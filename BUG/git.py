import gitlab
import re

# GitLab server URL and personal access token
GITLAB_URL = '' #your URL
PRIVATE_TOKEN = ''

# Initialize a GitLab connection
gl = gitlab.Gitlab(GITLAB_URL, private_token=PRIVATE_TOKEN)

def gitlab_extraction(gitlab_url):
    match = re.match(r'""""', gitlab_url)
    project_path = match.group(1)
    merge_request_id = int(match.group(2))

    project = gl.projects.get(project_path)
    merge_request = project.mergerequests.get(merge_request_id)
    changes = merge_request.changes()

    diff_text = "<div class='diff'>"
    for change in changes['changes']:
        diff = change['diff']
        diff_lines = diff.split('\n')
        diff_text += f"<h3>{change['new_path']}</h3><pre>"
        # diff_text +='\n'
        for line in diff_lines:
            if line.startswith('+'):
                diff_text += f"<span style='color: green;'>{line}</span><br>"
            elif line.startswith('-'):
                diff_text += f"<span style='color: red;'>{line}</span><br>"
            else:
                diff_text += f"{line}<br>"
        diff_text += "</pre>"
    diff_text += "</div>"
    return diff_text

def generate_html_page(gitlab_urls):
    print("entered here")
    def_content = "<html><body>"
    for url in gitlab_urls:
        def_content += gitlab_extraction(url)
    def_content += "</body></html>"

    with open("/home/Dhanush/DHANUSH_AI/templates/code_changes.html", "w") as file:
        file.write(def_content)
