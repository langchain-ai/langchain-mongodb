# Langchain MongoDB Releases

## Prep the JIRA Release

- Go to the release in [JIRA](https://jira.mongodb.org/projects/INTPYTHON?selectedItem=com.atlassian.jira.jira-projects-plugin%3Arelease-page&status=unreleased).

- Make sure there are no unfinished tickets. Move them to another version if need be.

- Click on the triple dot icon to the right and select "Edit".

- Update the description for a quick summary.

## Prep the Release

- Create a PR to bump the version and update the changelog, including today's date.
  Bump the minor version for new features, patch for a bug fix.

- Merge the PR.

## Run the Release Workflow

- Got to the release [workflow](https://github.com/langchain-ai/langchain-mongodb/actions/workflows/_release.yml).

- Click "Run Workflow".

- Choose the appropriate library from the dropdown.

- Click "Run Workflow".

- The workflow will create the tag, release to PyPI, and create the GitHub Release.

## JIRA Release

- Return to the JIRA release [list](https://jira.mongodb.org/projects/INTPYTHON?selectedItem=com.atlassian.jira.jira-projects-plugin%3Arelease-page&status=unreleased).

- Click "Save".

- Click on the triple dot again and select "Release".

- Enter today's date, and click "Confirm".

- Click "Release".


## Finish the Release

- Return to the release action and wait for it to complete successfully.

- Announce the release on Slack.  e.g "ANN: langchain-mongodb 0.5 with support for GraphRAG.
