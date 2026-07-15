# Contributing to OpenVINO™ Intel® NPU Compiler

We welcome contributions to the **OpenVINO™ Intel® NPU Compiler** project from both Intel employees and the open-source community.  
This document provides guidelines to help you make successful contributions and ensure a smooth review and integration process.
If you want to contribute to OpenVINO NPU plugin or have general questions about OpenVINO Toolkit contributions, refer to the [OpenVINO Contributing Guide](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md)

## Table of Contents
1. [Ways to Contribute](#ways-to-contribute)  
2. [Code Contribution Process](#code-contribution-process)  
3. [Pull Request Requirements](#pull-request-requirements)  
4. [Review Process](#review-process)  
5. [Responsibilities](#responsibilities)  
6. [Additional Resources](#additional-resources)


## Ways to Contribute

### Report Bugs or Propose Features

- If you experience unexpected behavior or have ideas for improvement:
  - Submit a [GitHub issue](https://github.com/openvinotoolkit/npu_compiler/issues) with a clear description, logs, and steps to reproduce.
  - Or join the OpenVINO [GitHub Discussions](https://github.com/openvinotoolkit/openvino/discussions) if it's exploratory or architectural.

### Suggest Architecture Improvements

- For major proposals or POCs, open a **Draft Pull Request** and mark it as such to initiate discussion without triggering full review flow.

### Contribute Code

- Start from a [Good First Issue](https://github.com/orgs/openvinotoolkit/projects/3) or any open item in the issue tracker.
- Follow our [Pull Request Process](#code-contribution-process).


## Code Contribution Process

The OpenVINO™ Intel® NPU Compiler project inherits its contribution process from the main [OpenVINO PR Contributing Guidelines](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_PR.md). We recommend familiarizing yourself with them before submitting a PR.

1. **Fork the Repository**  
   All contributions must come from **forks**, not branches in the main repository. This helps streamline CI and maintain stability.

2. **Create a Draft PR for Early Feedback**  
   - Draft PRs will **not auto-assign codeowners**.  
   - Use this to share ideas, work-in-progress or early architecture.

3. **Mark as Ready for Review**  
   - Remove "Draft" status  
   - Apply `READY_FOR_REVIEW` label  
   - Follow [PR Requirements](#pull-request-requirements)

4. **Get Review and Approval**  
   - Request reviews from relevant [Codeowners](./CODEOWNERS)  
   - Address feedback, rebase your branch, and rerun validation

5. **Ready for Merge**  
   - Apply `READY_FOR_MERGE` label  
   - Assign to a [Maintainer](#maintainer) once validation passes and approvals are in place


## Pull Request Requirements

Your PR must:

✅ **Target a single purpose (Feature / Bug / Refactor / etc.)**  
✅ Be tied to a JIRA ticket or GitHub issue with a clear description and acceptance criteria  
✅ Include a meaningful title, summary, and platform/classification  
✅ Be rebased on the target branch (no merge commits)  
✅ Contain logically separated commits with descriptions (avoid "squash-all" style)  
✅ Add test coverage for all new or affected behavior  
✅ Use proper GitHub labels, milestones, and assignees  
✅ Contain **validation results**, either:
   - Automatic CI logs
   - Links to manual test runs if auto-checks are not applicable  

## Review Process

- Request initial review from subject matter experts
- Make sure **all discussions are resolved**, especially blocking ones
- PR can only be assigned to a Maintainer once:
  - All codeowners have approved  
  - CI is green or justified in PR comments  
  - Discussions are closed  
- Maintainer performs final review and merge

> **Note:** For large PRs (e.g. refactor, submodule bump), mark it explicitly in the header so maintainers can manage merge conflicts accordingly.
## Responsibilities

### Pull Request Author
- Own the PR lifecycle from creation to merge
- Rebase, resolve comments, and ensure validation
- Escalate blockers proactively
- Ensure functionality lands in the correct branches

### Reviewer
- Provide timely, constructive feedback
- Approve only when confident in change scope and correctness

### Maintainer
- Confirm all merge conditions are satisfied
- Avoid conflicts with other recent changes
- Merge only when CI + review criteria are met

## Additional Resources

- [OpenVINO Code Style Guide](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/coding_style.md)  
- [How to fork a repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo)  
- [CODEOWNERS](./CODEOWNERS)

💬 **Got Questions?**  
Submit a [GitHub issue](https://github.com/openvinotoolkit/npu_compiler/issues) or start a [GitHub Discussion](https://github.com/openvinotoolkit/openvino/discussions) or reach out to the development team through internal channels (for Intel employees).


---

By contributing to this repository, you agree that your work will be licensed under the terms of the [LICENSE](./LICENSE).