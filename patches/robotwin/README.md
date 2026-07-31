# RoboTwin patches

`0001-pin-hopper-safe-curobo.patch` is applied to the RoboTwin submodule at
build or setup time. It pins every RoboTwin CuRobo installation path to
`gpulost/curobo@3490ef46d4ffbbf6756b91fb13b68215828533b0`, which contains the
Hopper-safe LBFGS warp reduction proposed in
https://github.com/NVlabs/curobo/pull/708.

The patch currently targets RoboTwin commit
`daef39a2f43226fb5af87552544e03d1f1bc70d9`. Run both `git apply --check`
against the submodule and the Robotwin2 H20 smoke test when updating either
repository revision.
