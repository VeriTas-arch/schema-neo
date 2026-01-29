"""
生成一个包含 figure、model/temp、src 以及自动生成的 Markdown+PDF 报告的发布目录。

功能：
1. 提示用户确认 `figure/` 内容为最新（可用 --yes 跳过）
2. 收集 `figure/` 下图片并从 `src/generator/` 中提取脚本开头的说明文字
3. 生成 `report.md`（放入发布目录根），并在可用时用 pandoc 生成 `report.pdf`
4. 将 `figure/`, `model/temp/`, `src/` 复制到以时间戳或版本命名的发布目录

"""

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure"
GENERATOR_DIR = ROOT / "src" / "generator"
MODEL_TEMP_DIR = ROOT / "model" / "temp"
RELEASE_DIR = ROOT / "release"


# User-editable configuration (replace CLI with variable control)
# Edit these variables to control the script instead of using command-line args.
CFG = {
    # release name (None -> timestamp)
    "name": None,
    # skip confirmation prompt
    "yes": False,
    # dry-run mode
    "dry_run": False,
}


def confirm_latest(skip: bool) -> bool:
    if skip:
        print("跳过检查：已由命令行参数确认 figure 最新。")
        return True
    print(
        "请确认 'figure/' 目录下的结果是最新的。若需要继续，请按回车；若取消请 Ctrl-C。"
    )
    try:
        input()
    except KeyboardInterrupt:
        print("\n已取消")
        return False
    return True


def collect_figures(fig_dir: Path) -> List[Path]:
    """Collect top-level figure files under `figure/` (not recursive).

    This is kept for backward compatibility / unmatched images listing.
    """
    exts = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    if not fig_dir.exists():
        return []
    figs = [
        p for p in sorted(fig_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()
    ]
    return figs


def collect_figures_in_subdir(subdir: Path) -> List[Path]:
    """Recursively collect image files under a specific figure subdirectory."""
    exts = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    if not subdir.exists():
        return []
    figs: List[Path] = []
    for p in sorted(subdir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            figs.append(p)
    return figs


def extract_docstring_or_comments(script_path: Path) -> str:
    """Try to extract module-level docstring or leading comments from a python script.

    If the file is not python or has no docstring, try to collect leading # comments.
    """
    try:
        text = script_path.read_text(encoding="utf-8")
    except Exception:
        return ""

    # try module docstring
    stripped = text.lstrip()
    if stripped.startswith(('"""', "'''")):
        quote = stripped[:3]
        end = stripped.find(quote, 3)
        if end != -1:
            return stripped[3:end].strip()

    # fallback: leading # comments
    lines = text.splitlines()
    comments = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("#"):
            comments.append(s.lstrip("#").strip())
        elif s == "":
            # allow one blank line between comments and code
            if comments:
                break
            else:
                continue
        else:
            break
    return "\n".join(comments).strip()


def collect_scripts(generator_dir: Path) -> Dict[str, str]:
    """Collect python scripts and extract their leading docs."""

    docs: Dict[str, str] = {}
    if not generator_dir.exists():
        return docs
    for p in sorted(generator_dir.iterdir()):
        if not (p.is_file() and p.suffix == ".py"):
            continue

        name = p.name
        # 强制规则：只处理以 subspace_ 开头的脚本
        if not name.startswith("subspace_"):
            print(f"跳过脚本 (非 subspace_ 前缀): {name}")
            continue
        skip = False
        if skip:
            print(f"跳过脚本 (匹配排除模式): {name}")
            continue

        doc = extract_docstring_or_comments(p)
        docs[name] = doc
    return docs


def generate_markdown(
    report_path: Path,
    script_docs: Dict[str, str],
    fig_dir: Path,
    fig_rel_path: str = "figure",
) -> None:
    lines: List[str] = []
    lines.append("# 自动生成分析报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().isoformat()}")
    lines.append("")

    # helpers to normalize titles and docstring headings
    def human_title_from_filename(fname: str) -> str:
        s = Path(fname).stem
        if s.startswith("subspace_"):
            s = s[len("subspace_") :]
        s = s.replace("_", " ").strip()
        return s[:1].upper() + s[1:] if s else fname

    def normalize_docstring_to_md(doc: str) -> List[str]:
        out: List[str] = []
        if not doc:
            return out
        lines_doc = doc.splitlines()
        i = 0
        while i < len(lines_doc):
            ln = lines_doc[i]
            # underlined heading: next line all - or =, length >=3
            if (
                i + 1 < len(lines_doc)
                and set(lines_doc[i + 1].strip()) <= set("-=")
                and len(lines_doc[i + 1].strip()) >= 3
            ):
                out.append(f"### {ln.strip()}")
                i += 2
                continue
            # single word headings like Summary or Notes
            low = ln.strip().lower().rstrip(":")
            if low in {"summary", "notes", "conclusion", "note"}:
                out.append(f"### {ln.strip().rstrip(':')}")
                i += 1
                continue
            out.append(ln)
            i += 1
        return out

    # For each script, include doc and related figures
    for script, doc in script_docs.items():
        title = human_title_from_filename(script)
        lines.append(f"## {title}")
        lines.append("")
        if doc:
            normalized = normalize_docstring_to_md(doc)
            if normalized:
                for line in normalized:
                    lines.append(line)
                lines.append("")
            else:
                lines.append(doc)
                lines.append("")
        else:
            lines.append("_未找到说明文字_")
            lines.append("")

        # Determine expected figure subdirectory based on script name.
        # Follows logic in src/generator/utils.initialize_analysis: scripts like
        # 'subspace_{name}.py' place figures under figure/{name}/
        stem = Path(script).stem
        extracted_name = None
        if script.startswith("subspace_") and script.endswith(".py"):
            # strip prefix and suffix
            extracted_name = script[len("subspace_") : -len(".py")]
        # fallback to stem
        subdir_name = extracted_name or stem
        subdir = fig_dir / subdir_name

        related = collect_figures_in_subdir(subdir)
        if related:
            for f in related:
                # compute relative path from report location: figure/{subdir}/{file}
                rel = f"{fig_rel_path}/{subdir_name}/{f.name}"
                lines.append(f"![{f.name}]({rel})")
                # add caption (image filename) below the image
                lines.append(f"*图：{f.name}*")
                lines.append("")
        else:
            # fallback: try matching by stem in top-level figure files
            top_level = collect_figures(fig_dir)
            fallback = [f for f in top_level if stem in f.name]
            if fallback:
                for f in fallback:
                    rel = f"{fig_rel_path}/{f.name}"
                    lines.append(f"![{f.name}]({rel})")
                    lines.append(f"*图：{f.name}*")
                    lines.append("")
            else:
                lines.append("_未找到相关图像_")
                lines.append("")

    # Also include any figures not matched
    # Also include any top-level figures (not in subfolders) that were not matched
    top_level_figs = collect_figures(fig_dir)
    matched_top = {
        f
        for script in script_docs
        for f in top_level_figs
        if Path(script).stem in f.name
    }
    unmatched = [f for f in top_level_figs if f not in matched_top]
    if unmatched:
        lines.append("## 其他图像")
        lines.append("")
        for f in unmatched:
            rel = f"{fig_rel_path}/{f.name}"
            lines.append(f"![{f.name}]({rel})")
            lines.append(f"*图：{f.name}*")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成 Markdown 报告: {report_path}")


def convert_markdown_to_typst(md_path: Path, typst_path: Path) -> None:
    """Convert a markdown file to Typst source using pandoc."""
    pandoc_exe = shutil.which("pandoc")
    if not pandoc_exe:
        raise FileNotFoundError(
            "pandoc not found in PATH. Please install pandoc to enable typst conversion."
        )

    typst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [pandoc_exe, str(md_path), "-o", str(typst_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Unknown error"
        raise RuntimeError(f"pandoc conversion failed: {stderr}")

    print(f"Generated typst source at {typst_path}")


def copy_into_release(
    release_dir: Path, fig_dir: Path, model_temp: Path, src_dir: Path
) -> None:
    # copy figure
    if fig_dir.exists():
        dst_fig = release_dir / "figure"
        shutil.copytree(fig_dir, dst_fig)
        print(f"复制 figure -> {dst_fig}")
    else:
        print("未找到 figure 目录。跳过复制。")

    # copy model/temp
    if model_temp.exists():
        dst_model = release_dir / "model" / "temp"
        dst_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_temp, dst_model)
        print(f"复制 model/temp -> {dst_model}")
    else:
        print("未找到 model/temp。跳过复制。")

    # copy src
    if src_dir.exists():
        dst_src = release_dir / "src"
        shutil.copytree(src_dir, dst_src)
        print(f"复制 src -> {dst_src}")
    else:
        print("未找到 src。跳过复制。")

    # copy root changelog if present
    changelog_src = ROOT / "CHANGELOG.md"
    if changelog_src.exists():
        try:
            shutil.copy2(changelog_src, release_dir / "CHANGELOG.md")
            print(f"复制 CHANGELOG.md -> {release_dir / 'CHANGELOG.md'}")
        except Exception as e:
            print(f"复制 CHANGELOG.md 失败: {e}")


def make_release(name: Optional[str], skip_confirm: bool, dry_run: bool) -> Path:
    if not confirm_latest(skip_confirm):
        raise SystemExit(1)

    figures = collect_figures(FIG_DIR)
    script_docs = collect_scripts(GENERATOR_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    release_name = name or timestamp
    release_dir = RELEASE_DIR / f"release_{release_name}"

    if dry_run:
        print("dry-run 模式，以下操作不会实际写入磁盘:")
        print(f" - 计划生成目录: {release_dir}")
        print(f" - 将收集 {len(figures)} 个图像，{len(script_docs)} 个脚本说明")
        return release_dir

    if release_dir.exists():
        print(f"发布目录已存在，删除后重新创建: {release_dir}")
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    # generate report.md inside release dir
    md_path = release_dir / "report.md"
    generate_markdown(md_path, script_docs, FIG_DIR, fig_rel_path="figure")

    # convert markdown to typst source using pandoc
    typst_path = release_dir / "report.typ"
    convert_markdown_to_typst(md_path, typst_path)

    # copy folders
    copy_into_release(release_dir, FIG_DIR, MODEL_TEMP_DIR, ROOT / "src")

    print("\n发布已生成:")
    print(f" - 目录: {release_dir}")
    print(f" - Markdown 报告: {md_path}")
    print(f" - Typst 报告: {typst_path}")

    return release_dir


def main() -> int:
    try:
        make_release(CFG.get("name"), CFG.get("yes"), CFG.get("dry_run"))
        return 0
    except Exception as e:
        print(f"错误: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
