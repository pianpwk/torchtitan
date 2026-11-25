# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.components.optimizer.optimizer import build_dist_muon
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    dist_muon,
    Owned,
)
from torchtitan.distributed.flex_shard import _optimizer_reshard_schedule
from torchtitan.distributed.flex_shard._optimizer_reshard_runtime import (
    _BucketedRedistributionRuntime,
)


class TestMuonPlanConstruction(unittest.TestCase):
    def test_identical_layers_build_one_plan(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        num_layers = 3
        num_matrices = 3
        matrix_rows = matrix_cols = 4
        names = [f"layers.{layer}.weight" for layer in range(num_layers)]
        # Six-row storage shards split the middle four-row matrix, requiring
        # the same redistribution plan for each layer.
        params = [
            torch.nn.Parameter(
                DTensor.from_local(
                    torch.empty(num_matrices * matrix_rows // 2, matrix_cols),
                    mesh,
                    (Shard(0),),
                    shape=torch.Size((num_matrices * matrix_rows, matrix_cols)),
                    stride=(matrix_cols, 1),
                )
            )
            for _ in names
        ]

        with (
            # CPU stream setup does not support the runtime's device argument.
            patch.object(_BucketedRedistributionRuntime, "reserve_buffers"),
            patch.object(
                dist_muon,
                "_build_region_redistribution_plan",
                wraps=dist_muon._build_region_redistribution_plan,
            ) as build_plan,
        ):
            build_dist_muon(
                [{"params": params, "param_names": names}],
                compute_sharding_by_fqn={
                    name: ComputeLayout(
                        {"dp_shard": BlockShard(dim=0, block_sizes=(matrix_rows,))},
                        compute_dtype=torch.bfloat16,
                    )
                    for name in names
                },
                bucket_configs=[BucketConfig(patterns=(name,)) for name in names],
            )
        build_plan.assert_called_once()


class TestEqualAllToAllSplits(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_equal_splits_match_packed_updates(self):
        # Owned matrices of different sizes: each owner receives a whole
        # matrix, so packed splits differ per (source, destination) pair.
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("dp_shard",))
        shapes = {"layers.0.a": (8, 6), "layers.0.b": (12, 5), "layers.1.c": (4, 7)}
        torch.manual_seed(0)
        values = {name: torch.randn(shape) for name, shape in shapes.items()}
        grads = {name: torch.randn(shape) for name, shape in shapes.items()}
        layout = ComputeLayout({"dp_shard": Owned()}, compute_dtype=torch.bfloat16)

        def step(equal_splits: bool) -> tuple[list[torch.Tensor], list[list[int]]]:
            params = [
                torch.nn.Parameter(distribute_tensor(values[name], mesh, (Shard(0),)))
                for name in shapes
            ]
            split_sizes = []
            all_to_all = dist.all_to_all_single

            def record(output, input, output_split_sizes, input_split_sizes, **kw):
                split_sizes.append(list(input_split_sizes))
                return all_to_all(
                    output, input, output_split_sizes, input_split_sizes, **kw
                )

            equal = patch.object(
                _optimizer_reshard_schedule,
                "_requires_equal_all_to_all_splits",
                return_value=equal_splits,
            )
            equal_owner = patch.object(
                dist_muon,
                "_requires_equal_all_to_all_splits",
                return_value=equal_splits,
            )
            with equal, equal_owner, patch.object(dist, "all_to_all_single", record):
                optimizer = build_dist_muon(
                    [{"params": params, "param_names": list(shapes)}],
                    compute_sharding_by_fqn=dict.fromkeys(shapes, layout),
                    bucket_configs=[BucketConfig(patterns=tuple(shapes))],
                    lr=0.1,
                )
                for param, name in zip(params, shapes, strict=True):
                    param.grad = distribute_tensor(grads[name], mesh, (Shard(0),))
                optimizer.step()
            return [param.full_tensor() for param in params], split_sizes

        packed, packed_splits = step(False)
        padded, padded_splits = step(True)
        self.assertTrue(any(len(set(s)) > 1 for s in packed_splits))
        self.assertTrue(padded_splits)
        self.assertTrue(all(len(set(s)) == 1 for s in padded_splits))
        for expected, actual in zip(packed, padded, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
