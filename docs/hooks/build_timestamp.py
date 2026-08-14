import datetime

def on_page_markdown(markdown, page, config, files):
    if page.file.src_path == "index.md" or page.file.src_uri == "index.md":
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        replacement = (
            f"\n!!! info \"Build Information\"\n"
            f"    **Documentation last built:** `{now_utc}`\n"
        )
        if "<!-- BUILD_TIMESTAMP -->" in markdown:
            markdown = markdown.replace("<!-- BUILD_TIMESTAMP -->", replacement)
        else:
            markdown += f"\n\n{replacement}\n"
    return markdown
