const PACKAGE_LABELS = new Map([
  ["### Package\n\nlangchain-mongodb\n", "pkg:langchain-mongodb"],
  ["### Package\n\nlanggraph-checkpoint-mongodb\n", "pkg:langgraph-checkpoint"],
  ["### Package\n\nlanggraph-store-mongodb\n", "pkg:langgraph-store"],
  ["### Package\n\nlangchain-mongodb-deepagents-vfs\n", "pkg:deepagents-vfs"],
  ["### Package\n\nCross-package / unsure\n", null],
]);
export default async function labelIssuePackage({ github, context }) {
  const body = (context.payload.issue.body || "").replaceAll("\r\n", "\n");
  const packageLabel = [...PACKAGE_LABELS].find(([marker]) => body.includes(marker))?.[1];
  if (!packageLabel) return;

  await github.rest.issues.addLabels({
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: context.issue.number,
    labels: [packageLabel],
  });
}