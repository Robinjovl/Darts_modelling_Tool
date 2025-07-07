#pip install GitPython
import git
from openpyxl import Workbook

# works from current folder, so synchronize it first
repo = git.Repo(".")

ver = 'v1.3.0'  # previous release version
branch = 'development'  #  branch for processing

# commit which corresponds to previous release
end_commit = repo.commit(repo.tags[ver])

wb = Workbook()
ws = wb.active  # get active worksheet

commits_list = list(repo.iter_commits(branch))
# loop by commits, fill the worksheet
for commit in commits_list:
    # columns of worksheet: date, author, commit_message
    commit_data = []
    commit_data.append(commit.committed_datetime.date())
    commit_data.append(commit.author.name)
    commit_data.append(commit.message)
    if commit == end_commit:
        break
    ws.append(commit_data)  # add row to worksheet

# Save the file
wb.save("whatsnew.xlsx")
