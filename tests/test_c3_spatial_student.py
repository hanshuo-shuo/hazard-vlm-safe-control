"""Cross-library exposure consistency and evaluation-metric semantics."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from c3_safe.geometry import GroundProjection, measure_exposure
from c3_safe.spatial_student import SpatialRequirementNet
from scripts.train_c3_spatial_baseline import average_precision, roc_auc, calibration_error


def test_torch_training_exposure_agrees_with_numpy_label_pipeline():
    field = np.random.default_rng(9).uniform(0,1,(32,32,3)).astype(np.float32)
    projection = GroundProjection.orthographic(32,32,(-2,2,-2,2))
    target,mask = measure_exposure(field,[[-1.,0.],[.6,.4]],.3,projection)
    tensor = torch.tensor(field.transpose(2,0,1),requires_grad=True)
    prediction = torch.quantile(tensor[:,torch.from_numpy(mask)],.95,dim=1)
    np.testing.assert_allclose(prediction.detach().numpy(),target.values,atol=1e-6)
    prediction.sum().backward()
    assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum()>0


def test_tied_metric_scores_are_order_independent_and_missing_classes_are_explicit():
    assert average_precision([1,0,0],[.5,.5,.5]) == pytest.approx(1/3)
    assert average_precision([0,0,1],[.5,.5,.5]) == pytest.approx(1/3)
    assert roc_auc([1,0],[.5,.5]) == .5
    assert average_precision([0,0],[.2,.3]) is None
    assert roc_auc([0,0],[.2,.3]) is None
    assert calibration_error([0,1],[0,1]) == 0
    assert calibration_error([0,0],[.9,.9]) == pytest.approx(.9)


def test_saved_rgb_student_reloads_without_optimizer_or_simulator(tmp_path):
    torch.manual_seed(8)
    model = SpatialRequirementNet().eval()
    image = torch.rand(2,3,31,47)
    expected = model.predict_field(image)
    path = tmp_path/"model.pt"
    torch.save(model.state_dict(),path)
    restored = SpatialRequirementNet().eval()
    restored.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
    assert torch.equal(restored.predict_field(image),expected)
    assert expected.shape == image.shape and torch.isfinite(expected).all()
