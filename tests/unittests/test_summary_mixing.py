import torch
import torch.nn


def test_summary_mixing_shape(device):

    from speechbrain.nnet.summary_mixing import SummaryMixing

    torch.manual_seed(666)

    batch = 8
    time = 10
    fea = 64

    inputs = torch.rand(batch, time, fea, device=device)

    # standard SummaryMixing
    sm_layer_mono_head = SummaryMixing(
        enc_dim=fea,
        nhead=1,
        local_proj_hid_dim=[32],
        local_proj_out_dim=32,
        summary_out_dim=fea,
        mode="SummaryMixing",
    )
    sm_layer_multi_heads = SummaryMixing(
        enc_dim=fea,
        nhead=4,
        local_proj_hid_dim=[32],
        local_proj_out_dim=32,
        summary_out_dim=fea,
        mode="SummaryMixing",
    )

    assert sm_layer_mono_head(inputs).shape == (batch, time, fea)
    assert sm_layer_multi_heads(inputs).shape == (batch, time, fea)

    # SummaryMixing-lite
    sm_layer_mono_head = SummaryMixing(
        enc_dim=fea,
        nhead=1,
        local_proj_hid_dim=[32],
        local_proj_out_dim=32,
        summary_out_dim=fea,
        mode="SummaryMixing-lite",
    )
    sm_layer_multi_heads = SummaryMixing(
        enc_dim=fea,
        nhead=4,
        local_proj_hid_dim=[32],
        local_proj_out_dim=32,
        summary_out_dim=fea,
        mode="SummaryMixing-lite",
    )

    assert sm_layer_mono_head(inputs).shape == (batch, time, fea)
    assert sm_layer_multi_heads(inputs).shape == (batch, time, fea)


def test_summary_mixing_value(device):

    from speechbrain.nnet.summary_mixing import SummaryMixing

    torch.manual_seed(666)

    fea = 4

    inputs = torch.tensor(
        [
            [[0.7527, 0.3651, 0.9038, 0.7875], [0.6034, 0.2692, 0.5403, 0.5255]],
            [[0.1031, 0.3663, 0.3871, 0.8302], [0.7560, 0.5056, 0.2302, 0.5153]],
        ],
        device=device,
        dtype=torch.float32,
    )

    # standard SummaryMixing
    sm_layer_mono_head = SummaryMixing(
        enc_dim=fea,
        nhead=1,
        local_proj_hid_dim=[32],
        local_proj_out_dim=32,
        summary_out_dim=fea,
        mode="SummaryMixing",
    )
    sm_layer_multi_heads = SummaryMixing(
        enc_dim=fea,
        nhead=2,
        local_proj_hid_dim=[32],
        local_proj_out_dim=32,
        summary_out_dim=fea,
        mode="SummaryMixing",
    )

    output_mono = torch.tensor(
        [
            [[-0.1356, 1.4845, 0.6271, -0.0425], [-0.1517, 1.4976, 0.5921, 0.0088]],
            [[-0.1389, 1.0110, 0.2934, -0.0205], [-0.0934, 0.9248, 0.3408, -0.0590]],
        ],
        device=device,
        dtype=torch.float32,
    )

    output_multi = torch.tensor(
        [
            [[-0.0252, -0.1224, 0.1157, 0.3226], [-0.0247, -0.1194, 0.1150, 0.3373]],
            [[-0.0215, -0.1295, 0.1184, 0.3549], [-0.0227, -0.1185, 0.1323, 0.3442]],
        ],
        device=device,
        dtype=torch.float32,
    )

    assert torch.allclose(sm_layer_mono_head(inputs), output_mono, atol=1e-04)
    assert torch.allclose(sm_layer_multi_heads(inputs), output_multi, atol=1e-04)

    # SummaryMixing-lite
    sm_layer_mono_head = SummaryMixing(
        enc_dim=fea,
        nhead=1,
        local_proj_hid_dim=[32],
        local_proj_out_dim=32,
        summary_out_dim=fea,
        mode="SummaryMixing-lite",
    )
    sm_layer_multi_heads = SummaryMixing(
        enc_dim=fea,
        nhead=2,
        local_proj_hid_dim=[32],
        local_proj_out_dim=32,
        summary_out_dim=fea,
        mode="SummaryMixing-lite",
    )

    output_mono = torch.tensor(
        [
            [[0.1336, -0.0514, -0.0124, 0.0757], [0.1336, -0.0514, -0.0124, 0.0757]],
            [[0.0843, -0.0540, -0.0040, 0.0419], [0.0843, -0.0540, -0.0040, 0.0419]],
        ],
        device=device,
        dtype=torch.float32,
    )

    output_multi = torch.tensor(
        [
            [[0.2983, -0.1313, 0.0977, 0.2732], [0.2983, -0.1313, 0.0977, 0.2732]],
            [[0.2979, -0.1314, 0.0993, 0.2745], [0.2979, -0.1314, 0.0993, 0.2745]],
        ],
        device=device,
        dtype=torch.float32,
    )

    assert torch.allclose(sm_layer_mono_head(inputs), output_mono, atol=1e-04)
    assert torch.allclose(sm_layer_multi_heads(inputs), output_multi, atol=1e-04)


test_summary_mixing_value("cpu")
test_summary_mixing_shape("cpu")
