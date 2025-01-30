import requests
from bs4 import BeautifulSoup

# Credentials
server_url = ''   #your required URL
username = ''
password = ''
accesstoken = ''
proxies = {
   """"""
}
headers = {
     'Content-Type': 'application/json',
     "Authorization": f"Bearer {accesstoken}"

 }

def fetch_issue_page(issue_key):
    url = f"{server_url}/rest/api/2/issue/{issue_key}/remotelink"
    response = requests.get(url, headers=headers)

   
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch issue page: {response.status_code}")
        # print("Response Body:", response.text)
        return None

def extract_gitlab_links(issue_page):
    gitlab_url = [data['object']['url'] for data in issue_page]
    return gitlab_url


def fetch_data(issue_ID):
    issue_key =  issue_ID # Replace with your issue key
    issue_page = fetch_issue_page(issue_key)
    if issue_page:
        gitlab_links = extract_gitlab_links(issue_page)
        print(f"GitLab links: {gitlab_links}")
        if gitlab_links:
            from git import generate_html_page
            code_changes = ''
            generate_html_page(gitlab_links)
            return ['generate_html_page(gitlab_links)']
        else:
            return []
    else:
        print("Gitlab link not found!!")
        import os
        if os.path.isfile("/home/Dhanush/DHANUSH_AI/templates/code_changes.html"):
            os.remove("/home/Dhanush/DHANUSH_AI/templates/code_changes.html")

fetch_data('ISSUE-46974')

