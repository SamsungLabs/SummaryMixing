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
