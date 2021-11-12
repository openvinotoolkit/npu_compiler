## Summary

(Please add a short summary why your pull request is useful)

## Target Platform For Release Notes (Mandatory)

Aids the generation of release notes [Explanation of Target Platform](https://wiki.ith.intel.com/display/Movidius/Explanation+of+Target+Platform+for+VPU+PR+Template)

- [ ] VPU2.0 (KMB)
- [ ] VPU2.0 (dKMB)
- [ ] VPU2.1 (TBH)
- [ ] VPU2.7 (MTL)
- [ ] VPU4 (LNL)
- [ ] NONE (Not included in release notes)

## Classification of this Pull Request

- [ ] Maintenance
- [ ] BUG
- [ ] Feature

## Related PRs

(Please add links to related PRs (if you have such PRs) and a small note why you depend on it)

* <pr-link> (<description>)

## Related tickets

(Please list tickets which the PR closes if you have any)

* https://jira.devtools.intel.com/browse/EISW-XXXXX

## CI

(Please replace the links below with your own)

#### Mandatory validation

(Default filter: `*precommit*:*smoke*`. Empty functional_tests filter for any major changes.)

* [ ] https://dsp-ci-icv.inn.intel.com/job/IE-MDK/job/manual/job/Ubuntu-Yocto/build
* [ ] https://dsp-ci-icv.inn.intel.com/job/IE-MDK/job/manual/job/Windows_dKMB/build

#### Validation for compiler changes / performance affected

(`*MLIR/precommit*` nets_included filter for VPUX compiler, `*MCM/precommit*` for MCM compiler.)

* [ ] https://dsp-ci-icv.inn.intel.com/job/Nets-Validation/job/manual/job/Yocto/build

#### Validation for dKMB focused changes in compiler or major changes

(Filters are the same as for Yocto.)

* [ ] https://dsp-ci-icv.inn.intel.com/job/Nets-Validation/job/manual/job/Windows/build

#### Compilation and single-image test Validation on moviSim for MTL related changes

(Default filter: `*MTL*`)

* [ ] https://dsp-ci-icv.inn.intel.com/job/Nets-Validation/job/manual/job/MoviSim/build

## Code Review Survey (Copy and Complete in your code review)
[Explanation of P1/P2/P3/P4 Defects](https://wiki.ith.intel.com/pages/viewpage.action?pageId=1684473024)

- number_minutes_spent_on_review[0]
- number_p1_defects_found[0]
- number_p2_defects_found[0]
- number_p3_defects_found[0]
- number_p4_defects_found[0]
