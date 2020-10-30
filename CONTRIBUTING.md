# How to Contribute

We welcome contributions at all levels: bugfixes, code improvements,
simplifications, new capabilities, improved documentation, new
examples/tutorials, etc.

Use a pull request (PR) toward the hippylib2muq:master branch to propose your
contribution. If you are planning significant code changes, or have any
questions, you should also open an
[issue](https://github.com/hippylib/hippylib2muq/issues) before issuing a PR.

`hIPPYlib-MUQ` is an interface program between `hIPPYlib` and `MUQ`;
contributions should be related to the interface; for contributing to each
program, please refer to their own repositories
([hIPPYlib](https://hippylib.github.io) and [MUQ](http://muq.mit.edu/)).


`hIPPYlib-MUQ` is maintained by https://github.com/hippylib, the main developer
hub for the hIPPYlib project.

All new contributions must be made under the terms of the [GPL3](./LICENSE)
license.

*By submitting a pull request, you are affirming the* [Developer's Certificate of
Origin](#developers-certificate-of-origin-11) *at the end of this file.*

## Quick Summary

- Please create development branches off `hippylib2muq:master`.
- Please follow the [developer guidelines](#developer-guidelines), in particular
  with regards to documentation and code styling.
- Pull requests should be issued toward `hippylib2muq:master`. Make sure
  to check the items off the [Pull Request Checklist](#pull-request-checklist).
- After approval, hIPPYlib-MUQ developers merge the PR in `hippylib2muq:master`.
- If you are also interested in the whole hIPPYlib project, we encourage you to
  join the hIPPYlib organization (see [here](https://github.com/hippylib/hippylib/blob/master/CONTRIBUTING.md#hippylib-organization)); if you are
  interested in the whole MUQ project, please take a look at [MUQ](http://muq.mit.edu).
- Don't hesitate to [contact us](#contact-information) if you have any questions.

## New Feature Development

- A new feature should be important enough that at least one person, the
  proposer, is willing to work on it and be its champion.

- The proposer creates a branch for the new feature (with suffix `-dev`), off
  the `master` branch, or another existing feature branch, for example:

  ```
  # Clone assuming you have setup your ssh keys on GitHub:
  git clone git@github.com:hippylib/hippylib2muq.git

  # Alternatively, clone using the "https" protocol:
  git clone https://github.com/hippylib/hippylib2muq.git

  # Create a new feature branch starting from "master":
  git checkout master
  git pull
  git checkout -b feature-dev

  # Work on "feature-dev", add local commits
  # ...

  # (One time only) push the branch to github and setup your local
  # branch to track the github branch (for "git pull"):
  git push -u origin feature-dev

  ```

- **We prefer that you create the new feature branch as a fork.**
  To allow hIPPYlib-MUQ developers to edit the PR, please [enable upstream edits](https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/).

- The typical feature branch name is `new-feature-dev`, e.g. `optimal_exp_design-dev`. While
  not frequent in `hippylib2muq`, other suffixes are possible, e.g. `-fix`, `-doc`, etc.


## Developer Guidelines

- *Keep the code lean and as simple as possible*
  - Well-designed simple code is frequently more general and powerful.
  - Lean code base is easier to understand by new collaborators.
  - New features should be added only if they are necessary or generally useful.
  - Code must be compatible with Python 3.
  - When adding new features add an example in the `example` or `application` folder and/or a
    new notebook in the `tutorial` folder.
  - The preferred way to export solutions for visualization in paraview is using `dl.XDMFFile`
  - The preferred way to save samples data is using [h5py](https://github.com/hippylib/hippylib2muq.git).

- *Keep the code general and reasonably efficient*
  - Main goal is fast prototyping for research.
  - When in doubt, generality wins over efficiency.
  - Respect the needs of different users (current and/or future).

- *Keep things separate and logically organized*
  - General usage features go in `hippylib2muq` (implemented in as much generality as
    possible), non-general features go into external apps/projects.
  - Inside `hippylib2muq`, compartmentalize between interface, mcmc, utility, etc.
  - Contributions that are project-specific or have external dependencies are
    allowed (if they are of broader interest), but should be `#ifdef`-ed and not
    change the code by default.

- Code specifics
  - All significant new classes, methods and functions have sphinx-style
    documentation in source comments.
  - Code styling should resemble existing code.
  - When manually resolving conflicts during a merge, make sure to mention the
    conflicted files in the commit message.

## Pull Requests

- When your branch is ready for other developers to review / comment on
  the code, create a pull request towards `hippylib2muq:master`.

- Pull request typically have titles like:

     `Description [new-feature-dev]`

  for example:

     `Bayesian Optimal Design of Experiments [oed-dev]`

  Note the branch name suffix (in square brackets).

- Titles may contain a prefix in square brackets to emphasize the type of PR.
  Common choices are: `[DON'T MERGE]`, `[WIP]` and `[DISCUSS]`, for example:

     `[DISCUSS] Bayesian Optimal Design of Experiments [oed-dev]`

- Add a description, appropriate labels and assign yourself to the PR. The hIPPYlib
  team will add reviewers as appropriate.

- List outstanding TODO items in the description.

<!-- - Track the Travis CI [continuous integration](#automated-testing) -->
<!--   builds at the end of the PR. These should run clean, so address any errors as -->
<!--   soon as possible. -->


## Pull Request Checklist

Before a PR can be merged, it should satisfy the following:

- [ ] Update `CHANGELOG`:
    - [ ] Is this a new feature users need to be aware of? New or updated application or tutorial?
    - [ ] Does it make sense to create a new section in the `CHANGELOG` to group with other related features?
- [ ] New examples/applications/tutorials:
    - [ ] All new examples/applications/tutorials run as expected.
    <!-- - [ ] Add a *fast version* of the example/application/tutorial to Travis CI -->
- [ ] New capability:
   - [ ] All significant new classes, methods and functions have sphinx-style documentation in source comments.
   - [ ] Add new examples/applications/tutorials to highlight the new capability.
   - [ ] For new classes, functions, or modules, edit the corresponding `.rst` file in the `doc` folder.
   - [ ] If this is a major new feature, consider mentioning in the short summary inside `README` *(rare)*.
   - [ ] If this is a `C++` extension, the `package_data` dictionary in `setup.py` should include new files.
<!-- - [ ] CI runs without errors. -->

## Contact Information

- Contact the hIPPYlib-MUQ team by posting to the [GitHub issue
  tracker](https://github.com/hippylib/hippylib2muq/issues).
  Please perform a search to make sure your question has not been answered already.

<!-- ## Slack channel -->
<!--  -->
<!-- The hIPPYlib slack channel is a good resource to request and receive help with using hIPPYlib. Everyone is invited to read and take part in discussions. Discussions about development of new features in hIPPYlib also take place here. You can join our Slack community by filling in [this form](https://forms.gle/w8B7uKSXxdVCmfZ99). -->

## [Developer's Certificate of Origin 1.1](https://developercertificate.org/)

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right
    to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my
    knowledge, is covered under an appropriate open source license and I have
    the right under that license to submit that work with modifications, whether
    created in whole or in part by me, under the same open source license
    (unless I am permitted to submit under a different license), as indicated in
    the file; or

(c) The contribution was provided directly to me by some other person who
    certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I
    submit with it, including my sign-off) is maintained indefinitely and may be
    redistributed consistent with this project or the open source license(s)
    involved.

---
> *Acknowledgement*: This file is made based on
[CONTRIBUTING.md](https://github.com/hippylib/hippylib/blob/master/CONTRIBUTING.md)
of [hIPPYlib](https://hippylib.github.io) which used [MFEM team](https://github.com/mfem) contributing
guidelines file as template.
