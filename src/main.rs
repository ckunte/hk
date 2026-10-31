use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

use lopdf::{dictionary, Document, Object, ObjectId};
use rayon::prelude::*;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

fn print_help() {
    print!(
r#"hk — housekeeping tools

usage: hk <command> [argument]

commands:
  cmbi   combine all PDF files in the current folder
  cmbs   combine PDF files per subfolder
  crop   crop PDF header/footer (--top/--bott N inches, default 1.0)
  fffp [--yes] [--all]
         fix permissions: dirs 755, files 644 (keeps +x, skips private)
  sffn   sanitise folder and file names
  help   show this help

author:
  Chetan Kunte, github.com/ckunte/hkp
"#
    );
}

/// "s" unless `n` is 1.
fn s(n: usize) -> &'static str { if n == 1 { "" } else { "s" } }

/// Case-insensitive extension check.
fn has_ext(path: &Path, ext: &str) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// Print collected per-item errors as warnings.
fn warn_all(errors: impl IntoIterator<Item = String>) {
    for e in errors { eprintln!("warning: {}", e); }
}

// ---------------------------------------------------------------------------
// Parallel filesystem walk
//
// Each directory's entries are read serially, but subdirectory recursion is
// dispatched to rayon's thread pool so all cores are used.  file_type() reads
// d_type from the readdir result — zero extra stat/lstat syscalls.
// ---------------------------------------------------------------------------

fn walk(root: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    // Pre-size to avoid repeated reallocations across a large tree.
    let dirs:  Mutex<Vec<(usize, PathBuf)>> = Mutex::new(Vec::with_capacity(512));
    let files: Mutex<Vec<PathBuf>>           = Mutex::new(Vec::with_capacity(4096));
    walk_inner(root, 0, &dirs, &files);
    let mut d = dirs.into_inner().unwrap();
    // Deepest dirs first so bottom-up rename is safe.
    // Sort by the depth integer we tracked during the walk — O(N log N) with
    // cheap integer comparisons instead of O(N log N × path_len) via
    // components().count() on every comparison.
    d.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let dirs = d.into_iter().map(|(_, p)| p).collect();
    (dirs, files.into_inner().unwrap())
}

fn walk_inner(
    dir: &Path,
    depth: usize,
    dirs:  &Mutex<Vec<(usize, PathBuf)>>,
    files: &Mutex<Vec<PathBuf>>,
) {
    // Iterate read_dir directly — avoids allocating a Vec<DirEntry> per
    // directory just to traverse it once.
    let Ok(rd) = fs::read_dir(dir) else { return };

    let mut local_files = Vec::new();
    let mut subdirs     = Vec::new();

    for entry in rd.flatten() {
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_symlink() { continue; }
        let path = entry.path();
        if ft.is_dir() {
            subdirs.push(path);
        } else if ft.is_file() {
            local_files.push(path);
        }
    }

    // One lock per directory, not one lock per file.
    files.lock().unwrap().extend(local_files);

    // Recurse into subdirectories in parallel across all cores.
    subdirs.par_iter().for_each(|sub| walk_inner(sub, depth + 1, dirs, files));

    // Record subdirs with their depth for the sort in walk().
    dirs.lock().unwrap().extend(subdirs.into_iter().map(|p| (depth + 1, p)));
}

/// PDFs directly inside `dir`, sorted, excluding a file named `skip` (the
/// combined output of a previous run, which must not be merged into itself)
/// and hidden files — notably macOS "._name.pdf" AppleDouble stubs, which
/// appear on non-HFS volumes and are not valid PDFs.
fn collect_sorted_pdfs(dir: &Path, skip: &str) -> Vec<PathBuf> {
    let Ok(rd) = fs::read_dir(dir) else { return vec![]; };
    let mut pdfs: Vec<PathBuf> = rd
        .flatten()
        .filter(|e| {
            e.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                && e.file_name() != skip
                && !e.file_name().to_string_lossy().starts_with('.')
                && has_ext(&e.path(), "pdf")
        })
        .map(|e| e.path())
        .collect();
    pdfs.sort_unstable();
    pdfs
}

fn dir_stem(dir: &Path) -> String {
    dir.canonicalize()
        .ok()
        .as_deref()
        .unwrap_or(dir)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("combined")
        .to_owned()
}

// ---------------------------------------------------------------------------
// fffp — fix permissions: dirs 755, files 644 (parallel chmod)
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn cmd_fffp(flags: &[String]) -> Result<()> {
    let assume_yes  = flags.iter().any(|a| a == "--yes" || a == "-y");
    let include_all = flags.iter().any(|a| a == "--all");

    let cwd = env::current_dir()?;
    let home = env::var_os("HOME").and_then(|h| PathBuf::from(h).canonicalize().ok());
    if cwd.parent().is_none() || home.as_deref() == Some(cwd.as_path()) {
        return Err("refusing to change permissions of / or your home folder".into());
    }

    println!("This will recursively set permissions under {}:", cwd.display());
    println!("  folders → 755, files → 644 (files that are already executable → 755)");
    if !include_all {
        println!("  private entries (no group/other access, e.g. 600 keys) are skipped; use --all to include them");
    }
    if !assume_yes {
        print!("Continue? [y/N] ");
        io::stdout().flush().ok();
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        if !matches!(answer.trim(), "y" | "Y" | "yes") {
            return Err("aborted".into());
        }
    }

    let (dirs, files) = walk(&cwd);
    let changed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());

    // Walk() never follows symlinks, so only real files and folders are touched.
    let apply = |path: &Path, is_dir: bool| {
        let mode = match fs::symlink_metadata(path) {
            Ok(m)  => m.permissions().mode(),
            Err(e) => { errors.lock().unwrap().push(format!("{}: {}", path.display(), e)); return; }
        };
        if !include_all && mode & 0o077 == 0 {
            skipped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let want = if is_dir || mode & 0o111 != 0 { 0o755 } else { 0o644 };
        if mode & 0o777 == want {
            return;
        }
        match fs::set_permissions(path, fs::Permissions::from_mode(want)) {
            Ok(_)  => { changed.fetch_add(1, Ordering::Relaxed); }
            Err(e) => { errors.lock().unwrap().push(format!("{}: {}", path.display(), e)); }
        }
    };
    apply(&cwd, true);
    dirs.par_iter().for_each(|d| apply(d, true));
    files.par_iter().for_each(|f| apply(f, false));

    let n = changed.load(Ordering::Relaxed);
    let sk = skipped.load(Ordering::Relaxed);
    println!("Changed {} entr{} ({} folder{} and {} file{} scanned).",
        n, if n == 1 { "y" } else { "ies" },
        dirs.len() + 1, s(dirs.len() + 1),
        files.len(),    s(files.len()));
    if sk > 0 {
        println!("Skipped {} private entr{}.", sk, if sk == 1 { "y" } else { "ies" });
    }
    let errors = errors.into_inner().unwrap();
    for e in errors.iter().take(10) {
        eprintln!("warning: {}", e);
    }
    if errors.len() > 10 {
        eprintln!("warning: … and {} more", errors.len() - 10);
    }
    if !errors.is_empty() {
        return Err(format!("{} permission change{} failed", errors.len(),
            s(errors.len())).into());
    }
    Ok(())
}

#[cfg(not(unix))]
fn cmd_fffp(_flags: &[String]) -> Result<()> {
    Err("fffp is not supported on this platform (Unix permission modes only)".into())
}

// ---------------------------------------------------------------------------
// sffn — sanitise folder and file names
//
// Rules:
//   • Any run of one or more non-alphanumeric characters (spaces, "&", "_",
//     "-", dots, punctuation, symbols, in any mix) becomes a single "-"
//   • Leading and trailing "-" are trimmed from each name or file stem
//   • Applies to directory names and to the stems of document and media files
//     (see SANITISE_EXTS); the extension is lowercased
//   • Compound archive extensions (.tar.gz, .tar.bz2, .tar.xz, ...) are kept
//     intact; only the stem before them is cleaned
//   • Dotfiles and dotfolders (leading dot), everything inside a dotfolder, and
//     other file types are left alone
//   • Case and accented letters are preserved
//
// File renames are parallelised; directory renames stay sequential
// (deepest-first) so parent paths are not invalidated mid-run.
// ---------------------------------------------------------------------------

/// File types whose names `sffn` sanitises (lowercase, no leading dot).
const SANITISE_EXTS: &[&str] = &[
    // documents
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp",
    "rtf", "txt", "md", "csv", "epub", "pages", "numbers", "key", "typ",
    // images
    "jpg", "jpeg", "png", "gif", "webp", "heic", "tif", "tiff", "bmp", "svg",
    // audio
    "mp3", "wav", "flac", "m4a", "aac", "ogg",
    // video
    "mp4", "mov", "mkv", "avi", "webm", "m4v",
];

/// Two-part archive extensions, kept intact (lowercased) instead of having the
/// inner dot turned into '-'.  Matched case-insensitively against the name's end.
const COMPOUND_EXTS: &[&str] = &[
    "tar.gz", "tar.bz2", "tar.xz", "tar.zst", "tar.lz", "tar.lzma", "tar.z",
];

/// Split a file name into (stem, lowercase extension) if `sffn` handles its
/// type: a compound archive extension first, otherwise the last extension if it
/// is listed in SANITISE_EXTS.  Returns None for anything else.
fn split_known_ext(name: &str) -> Option<(&str, String)> {
    // to_ascii_lowercase keeps byte lengths, so slicing `name` by it is safe.
    let lower = name.to_ascii_lowercase();
    for c in COMPOUND_EXTS {
        // Need at least one stem character before ".<ext>".
        let Some(cut) = name.len().checked_sub(c.len() + 1).filter(|&n| n > 0) else { continue };
        if lower.ends_with(c) && lower.as_bytes()[cut] == b'.' {
            return Some((&name[..cut], (*c).to_owned()));
        }
    }
    let dot = name.rfind('.').filter(|&i| i > 0 && i + 1 < name.len())?;
    let ext = lower[dot + 1..].to_owned();
    SANITISE_EXTS.contains(&ext.as_str()).then(|| (&name[..dot], ext))
}

/// Collapse every run of non-alphanumeric characters into a single '-', with
/// no leading or trailing '-'.  Returns an empty string if nothing is left.
fn collapse_separators(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_sep = false;
    for c in s.chars() {
        if c.is_alphanumeric() {
            if pending_sep && !out.is_empty() {
                out.push('-');
            }
            pending_sep = false;
            out.push(c);
        } else {
            pending_sep = true;
        }
    }
    out
}

fn sanitize_dir_name(name: &str) -> String {
    collapse_separators(name)
}

fn sanitize_file_stem(stem: &str) -> String {
    collapse_separators(stem)
}

/// Plan `from` -> `parent/new_name` only when it is safe: the new name must be
/// usable, must not already exist on disk, and must not be claimed by another
/// rename in this run.  `fs::rename` silently replaces an existing target on
/// Unix, so without this check "a b.pdf" and "a_b.pdf" would destroy one another.
fn plan_rename(
    from: &Path,
    new_name: &str,
    claimed: &mut HashSet<PathBuf>,
    skipped: &mut Vec<String>,
) -> Option<(PathBuf, PathBuf)> {
    let to = from.parent()?.join(new_name);
    if new_name.is_empty() || new_name.starts_with('.') {
        skipped.push(format!("{}: sanitised name '{}' is unusable", from.display(), new_name));
        return None;
    }
    if to.symlink_metadata().is_ok() || !claimed.insert(to.clone()) {
        skipped.push(format!("{}: target {} already exists", from.display(), to.display()));
        return None;
    }
    Some((from.to_path_buf(), to))
}

/// True if any component of `path` below `root` is a dotfile or dotfolder.
/// Such entries, and everything inside a dotfolder, are left alone by `sffn`.
fn in_hidden(path: &Path, root: &Path) -> bool {
    path.strip_prefix(root).unwrap_or(path).components().any(|c| {
        c.as_os_str().to_str().is_some_and(|n| n.starts_with('.'))
    })
}

fn cmd_sffn() -> Result<()> {
    let cwd = env::current_dir()?;
    let (dirs, files) = walk(&cwd);
    let count = AtomicUsize::new(0);
    let mut claimed: HashSet<PathBuf> = HashSet::new();
    let mut skipped: Vec<String> = Vec::new();

    // Plan every rename sequentially so collisions are detected
    // deterministically; only the (independent) execution is parallel.
    let mut file_plan: Vec<(PathBuf, PathBuf)> = Vec::new();
    for path in &files {
        if in_hidden(path, &cwd) { continue; }
        let Some(old_name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let Some((stem, ext)) = split_known_ext(old_name) else { continue };
        let new_name = format!("{}.{}", sanitize_file_stem(stem), ext);
        if new_name != old_name {
            if let Some(p) = plan_rename(path, &new_name, &mut claimed, &mut skipped) {
                file_plan.push(p);
            }
        }
    }

    // Directories are renamed deepest-first so parent paths stay valid.
    let mut dir_plan: Vec<(PathBuf, PathBuf)> = Vec::new();
    for path in &dirs {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        if in_hidden(path, &cwd) { continue; } // leave .git, .cache, ... and their subfolders alone
        let new_name = sanitize_dir_name(name);
        if new_name != name {
            if let Some(p) = plan_rename(path, &new_name, &mut claimed, &mut skipped) {
                dir_plan.push(p);
            }
        }
    }

    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    file_plan.par_iter().for_each(|(from, to)| {
        match fs::rename(from, to) {
            Ok(_)  => { count.fetch_add(1, Ordering::Relaxed); }
            Err(e) => { errors.lock().unwrap()
                            .push(format!("{}: {}", from.display(), e)); }
        }
    });
    for (from, to) in &dir_plan {
        match fs::rename(from, to) {
            Ok(_)  => { count.fetch_add(1, Ordering::Relaxed); }
            Err(e) => { errors.lock().unwrap()
                            .push(format!("{}: {}", from.display(), e)); }
        }
    }

    let n = count.load(Ordering::Relaxed);
    println!("Sanitised {} name{}.", n, s(n));
    warn_all(skipped.into_iter().chain(errors.into_inner().unwrap()));
    Ok(())
}

// ---------------------------------------------------------------------------
// PDF merge (cmbi / cmbs)
// PDF pages must be merged in document order, so this stays sequential.
// ---------------------------------------------------------------------------

fn type_of(obj: &Object) -> &str {
    obj.as_dict()
        .ok()
        .and_then(|d| d.get(b"Type").ok())
        .and_then(|t| t.as_name().ok())
        .and_then(|n| std::str::from_utf8(n).ok())
        .unwrap_or("")
}

/// Refuse absurdly large inputs (lopdf reads the whole file into memory).
const MAX_PDF_BYTES: u64 = 1 << 30;

/// Load a PDF, turning both parse errors and lopdf panics on malformed input
/// into an ordinary error naming the file.
fn load_pdf(path: &Path) -> Result<Document> {
    let len = fs::metadata(path)
        .map_err(|e| format!("{}: {}", path.display(), e))?
        .len();
    if len > MAX_PDF_BYTES {
        return Err(format!("{}: file too large ({} MiB, limit {} MiB)",
            path.display(), len >> 20, MAX_PDF_BYTES >> 20).into());
    }
    match std::panic::catch_unwind(|| Document::load(path)) {
        Ok(Ok(mut doc)) => {
            // Files with an owner-only password open with an empty user password.
            // Anything else would otherwise load as an empty document and be
            // silently written back as a near-empty file.
            if doc.is_encrypted() && doc.decrypt("").is_err() {
                return Err(format!("{}: password-protected PDF (not supported)",
                    path.display()).into());
            }
            Ok(doc)
        }
        Ok(Err(e))  => Err(format!("{}: {}", path.display(), e).into()),
        Err(_)      => Err(format!("{}: malformed PDF (parser crashed)", path.display()).into()),
    }
}

fn merge_pdfs(paths: &[PathBuf]) -> Result<Document> {
    let mut max_id: u32 = 1;
    let mut ordered_pages: Vec<(ObjectId, Object)> = Vec::new();
    let mut resources: BTreeMap<ObjectId, Object> = BTreeMap::new();

    for path in paths {
        let mut doc = load_pdf(path)?;
        doc.renumber_objects_with(max_id);
        max_id = doc.max_id + 1;

        let mut page_pairs: Vec<(u32, ObjectId)> =
            doc.get_pages().into_iter().collect();
        page_pairs.sort_by_key(|(n, _)| *n);
        for (_, page_id) in page_pairs {
            if let Some(obj) = doc.objects.get(&page_id) {
                ordered_pages.push((page_id, obj.clone()));
            }
        }

        for (id, obj) in &doc.objects {
            match type_of(obj) {
                "Page" | "Pages" | "Catalog" => {}
                _ => { resources.insert(*id, obj.clone()); }
            }
        }
    }

    let mut merged = Document::new();
    merged.version = "1.5".to_owned();
    let pages_id:   ObjectId = (max_id, 0);
    let catalog_id: ObjectId = (max_id + 1, 0);

    for (id, obj) in resources { merged.objects.insert(id, obj); }

    let page_count = ordered_pages.len() as i64;
    let mut kids: Vec<Object> = Vec::with_capacity(ordered_pages.len());
    for (id, mut page) in ordered_pages {
        if let Ok(dict) = page.as_dict_mut() {
            dict.set("Parent", Object::Reference(pages_id));
        }
        merged.objects.insert(id, page);
        kids.push(Object::Reference(id));
    }

    merged.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type"  => "Pages",
            "Kids"  => Object::Array(kids),
            "Count" => page_count,
        }),
    );
    merged.objects.insert(
        catalog_id,
        Object::Dictionary(dictionary! {
            "Type"  => "Catalog",
            "Pages" => Object::Reference(pages_id),
        }),
    );
    merged.trailer.set("Root", Object::Reference(catalog_id));
    merged.trailer.set("Size", (merged.objects.len() + 1) as i64);
    merged.compress();

    Ok(merged)
}

fn comb_name(dir: &Path) -> String {
    format!("{}-comb.pdf", dir_stem(dir))
}

/// Merge `pdfs` into `output`.  The result is written to a temporary file and
/// renamed into place, so a failed run never clobbers an earlier output.
fn write_combined(pdfs: &[PathBuf], output: &Path) -> Result<()> {
    let replacing = output.exists();
    let mut doc = merge_pdfs(pdfs)?;
    let tmp = output.with_file_name(format!(
        ".{}.tmp",
        output.file_name().unwrap().to_string_lossy()
    ));
    if let Err(e) = doc.save(&tmp) {
        let _ = fs::remove_file(&tmp);
        return Err(e.into());
    }
    fs::rename(&tmp, output).inspect_err(|_| { let _ = fs::remove_file(&tmp); })?;
    if replacing { print!("(replaced existing) "); }
    Ok(())
}

fn combine_dir(dir: &Path) -> Result<()> {
    let out_name = comb_name(dir);
    let pdfs = collect_sorted_pdfs(dir, &out_name);
    if pdfs.is_empty() { return Ok(()); }
    print!(
        "  {} ({} file{}) → {} ... ",
        dir_stem(dir),
        pdfs.len(),
        s(pdfs.len()),
        out_name
    );
    io::stdout().flush().ok();
    write_combined(&pdfs, &dir.join(&out_name))?;
    println!("done.");
    Ok(())
}

fn cmd_cmbi() -> Result<()> {
    let cwd = env::current_dir()?;
    if collect_sorted_pdfs(&cwd, &comb_name(&cwd)).is_empty() {
        println!("No PDF files found in the current directory.");
        return Ok(());
    }
    combine_dir(&cwd)
}

fn cmd_cmbs() -> Result<()> {
    let cwd = env::current_dir()?;
    let mut dirs: Vec<PathBuf> = fs::read_dir(&cwd)?
        .flatten()
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .map(|e| e.path())
        .collect();
    dirs.sort_unstable();

    if dirs.is_empty() {
        println!("No subdirectories found.");
        return Ok(());
    }

    println!("Combining PDFs in subdirectories of {}:", cwd.display());
    // One bad directory must not stop the rest; report failures at the end.
    let mut failures = 0usize;
    for dir in dirs {
        if let Err(e) = combine_dir(&dir) {
            println!("failed.");
            eprintln!("warning: {}", e);
            failures += 1;
        }
    }
    println!("Done.");
    if failures > 0 {
        return Err(format!("{} director{} failed", failures,
            if failures == 1 { "y" } else { "ies" }).into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// crop — trim PDF header/footer
// ---------------------------------------------------------------------------

fn obj_to_f64(obj: &Object) -> Result<f64> {
    match obj {
        Object::Integer(i) => Ok(*i as f64),
        Object::Real(f)    => Ok(*f as f64),
        _ => Err("expected a number in page box".into()),
    }
}

/// Look up a page attribute, walking up the /Parent chain for inherited keys
/// (MediaBox, CropBox, Rotate), and resolving indirect references.
fn inherited_attr(doc: &Document, page_id: ObjectId, key: &[u8]) -> Option<Object> {
    let mut id = page_id;
    for _ in 0..64 {
        let dict = doc.get_object(id).ok()?.as_dict().ok()?;
        if let Ok(v) = dict.get(key) {
            return match v {
                Object::Reference(r) => doc.get_object(*r).ok().cloned(),
                other => Some(other.clone()),
            };
        }
        id = dict.get(b"Parent").ok()?.as_reference().ok()?;
    }
    None
}

fn crop_one_pdf(path: &Path, top_pts: f64, bott_pts: f64) -> Result<PathBuf> {
    let mut doc = load_pdf(path)?;

    let mut pages: Vec<(u32, ObjectId)> = doc.get_pages().into_iter().collect();
    pages.sort_by_key(|(n, _)| *n);
    if pages.is_empty() {
        return Err(format!("{}: no pages found", path.display()).into());
    }

    for (page_num, page_id) in &pages {
        let obj = inherited_attr(&doc, *page_id, b"CropBox")
            .or_else(|| inherited_attr(&doc, *page_id, b"MediaBox"))
            .ok_or_else(|| format!("page {}: no MediaBox or CropBox found", page_num))?;
        let arr = obj.as_array()
            .map_err(|_| format!("page {}: box is not an array", page_num))?;
        if arr.len() < 4 {
            return Err(format!("page {}: box has fewer than 4 elements", page_num).into());
        }
        let resolve = |o: &Object| -> Result<f64> {
            match o {
                Object::Reference(r) => obj_to_f64(doc.get_object(*r)?),
                other => obj_to_f64(other),
            }
        };
        let (ax, ay, bx, by) = (resolve(&arr[0])?, resolve(&arr[1])?,
                                resolve(&arr[2])?, resolve(&arr[3])?);
        // Boxes may be given with corners in any order.
        let (x0, x1) = (ax.min(bx), ax.max(bx));
        let (y0, y1) = (ay.min(by), ay.max(by));

        // /Rotate turns the page clockwise when displayed, so "top" and
        // "bottom" as the viewer sees them are not always the y-edges.
        let rotate = match inherited_attr(&doc, *page_id, b"Rotate") {
            Some(Object::Integer(r)) => r,
            Some(Object::Real(r))    => r as i64,
            _ => 0,
        }.rem_euclid(360);

        let height = if rotate == 90 || rotate == 270 { x1 - x0 } else { y1 - y0 };
        if top_pts + bott_pts >= height {
            return Err(format!(
                "{}: page {}: combined crop ({:.2} + {:.2} = {:.2} pt) \
                 exceeds page height ({:.2} pt)",
                path.display(), page_num,
                top_pts, bott_pts, top_pts + bott_pts, height
            ).into());
        }

        // PDF y-axis is bottom-up. Visual top/bottom map to unrotated edges:
        //   0   : top = y1, bottom = y0
        //   90  : top = x0, bottom = x1
        //   180 : top = y0, bottom = y1
        //   270 : top = x1, bottom = x0
        let (nx0, ny0, nx1, ny1) = match rotate {
            90  => (x0 + top_pts, y0, x1 - bott_pts, y1),
            180 => (x0, y0 + top_pts, x1, y1 - bott_pts),
            270 => (x0 + bott_pts, y0, x1 - top_pts, y1),
            _   => (x0, y0 + bott_pts, x1, y1 - top_pts),
        };

        if let Some(obj) = doc.objects.get_mut(page_id) {
            if let Ok(dict) = obj.as_dict_mut() {
                dict.set("CropBox", Object::Array(vec![
                    Object::Real(nx0 as f32),
                    Object::Real(ny0 as f32),
                    Object::Real(nx1 as f32),
                    Object::Real(ny1 as f32),
                ]));
            }
        }
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let out  = path.with_file_name(format!("{}-cropped.pdf", stem));
    doc.compress();
    doc.save(&out)
        .map_err(|e| format!("{}: save failed: {}", out.display(), e))?;
    Ok(out)
}

fn cmd_crop(file_args: &[String]) -> Result<()> {
    let mut top:   f64          = 1.0;
    let mut bott:  Option<f64>  = None;
    let mut paths: Vec<PathBuf> = Vec::new();

    let mut i = 0;
    while i < file_args.len() {
        match file_args[i].as_str() {
            "--top" => {
                i += 1;
                top = file_args.get(i)
                    .ok_or("--top: missing value")?
                    .parse::<f64>()
                    .map_err(|_| "--top: value must be a number")?;
                if top < 0.0 { return Err("--top: must be non-negative".into()); }
            }
            "--bott" => {
                i += 1;
                let v = file_args.get(i)
                    .ok_or("--bott: missing value")?
                    .parse::<f64>()
                    .map_err(|_| "--bott: value must be a number")?;
                if v < 0.0 { return Err("--bott: must be non-negative".into()); }
                bott = Some(v);
            }
            arg => { paths.push(PathBuf::from(arg)); }
        }
        i += 1;
    }

    let bott = bott.unwrap_or(top);

    if paths.is_empty() {
        return Err(
            "crop: no PDF files specified.\n\
             Usage: hk crop [--top N] [--bott N] file.pdf [...]".into()
        );
    }

    let top_pts  = top  * 72.0;
    let bott_pts = bott * 72.0;

    println!(
        "Cropping {} file{} — top: {:.2}\", bottom: {:.2}\"",
        paths.len(), s(paths.len()),
        top, bott,
    );

    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let count = AtomicUsize::new(0);

    paths.par_iter().for_each(|path| {
        match crop_one_pdf(path, top_pts, bott_pts) {
            Ok(out) => {
                count.fetch_add(1, Ordering::Relaxed);
                println!("  {}", out.display());
            }
            Err(e) => {
                errors.lock().unwrap().push(e.to_string());
            }
        }
    });

    let n = count.load(Ordering::Relaxed);
    println!("Done: {}/{} file{} cropped.",
        n, paths.len(), s(n));
    warn_all(errors.into_inner().unwrap());
    Ok(())
}


// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let rest = args.get(2..).unwrap_or(&[]);

    let result: Result<()> = match cmd {
        "cmbi" => cmd_cmbi(),
        "cmbs" => cmd_cmbs(),
        "crop" => cmd_crop(rest),
        "fffp"  => cmd_fffp(rest),
        "sffn" => cmd_sffn(),
        "help" | "--help" | "-h" => { print_help(); Ok(()) }
        other => {
            eprintln!("hk: unknown command '{}'\n", other);
            print_help();
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod sffn_tests {
    use super::{sanitize_dir_name as d, sanitize_file_stem as f};

    #[test]
    fn ampersand_and_spaces() {
        assert_eq!(f("Tom & Jerry"), "Tom-Jerry");
        assert_eq!(f("a   &   b"), "a-b");
    }

    #[test]
    fn mixed_runs_collapse() {
        assert_eq!(f("foo_-_bar"), "foo-bar");
        assert_eq!(f("foo__bar"), "foo-bar");
        assert_eq!(f("foo---bar"), "foo-bar");
        assert_eq!(f("foo - bar (final)"), "foo-bar-final");
    }

    #[test]
    fn trims_ends() {
        assert_eq!(f("  -report_"), "report");
        assert_eq!(d("(draft)"), "draft");
    }

    #[test]
    fn dots_in_stems_and_dirs() {
        assert_eq!(f("v1.2.3 notes"), "v1-2-3-notes");
        assert_eq!(d("my.folder name"), "my-folder-name");
    }

    #[test]
    fn keeps_case_and_accents() {
        assert_eq!(f("Résumé 2026"), "Résumé-2026");
    }

    #[test]
    fn nothing_left_is_empty() {
        assert_eq!(f("&&&"), "");
        assert_eq!(d("___"), "");
    }

    #[test]
    fn compound_extensions_kept() {
        use super::split_known_ext as k;
        assert_eq!(k("My Backup.tar.gz"), Some(("My Backup", "tar.gz".to_string())));
        assert_eq!(k("data v1.2.TAR.XZ"), Some(("data v1.2", "tar.xz".to_string())));
        assert_eq!(k("a.b.pdf"), Some(("a.b", "pdf".to_string())));
        assert_eq!(k("plain.gz"), None);      // single .gz is not a handled type
        assert_eq!(k(".tar.gz"), None);       // no stem
        assert_eq!(k("tar.gz"), None);        // no stem
        assert_eq!(k("noext"), None);
    }
}
