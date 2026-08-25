# pip install GitPython
import git
from openpyxl import Workbook

# works from current folder, so synchronize it first
repo = git.Repo(".")

ver = 'v1.5.1'  # previous release version
branch = 'development'  #  branch for processing

# commit which corresponds to previous release
end_commit = repo.commit(repo.tags[ver])

print('end_commit', end_commit, end_commit.committed_datetime.date())

wb = Workbook()
ws = wb.active  # get active worksheet

commits_list = list(repo.iter_commits(branch))

last_commit_found = False
# loop by commits, fill the worksheet
for commit in commits_list:
    # columns of worksheet: date, author, commit_message
    commit_data = []
    commit_data.append(commit.committed_datetime.date())
    commit_data.append(commit.author.name)
    commit_data.append(commit.message)
    if commit == end_commit:
        last_commit_found = True
        break
    ws.append(commit_data)  # add row to worksheet

if not last_commit_found:
    print('Warning: end_commit was not found! Check the', branch, 'has a tag', ver)

# Save the file
wb.save("whatsnew.xlsx")
