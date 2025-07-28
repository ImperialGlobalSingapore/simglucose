import logging
from tqdm import tqdm
from pathlib import Path


from simglucose.patient.t1dm_patient import T1DMPatient, Action
import sys

sys.path.append(str(Path(__file__).parent.parent))
from test_utils import plot_and_show, plot_and_save, get_rmse

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# current file path
file_path = Path(__file__).resolve()
parent_folder = file_path.parent

img_dir = parent_folder / "imgs"
best_basal_folder = parent_folder / "results" / "best_basal_rate"


def test_patient(
    patient_name="adolescent#003",
    fig_title=None,
    save_fig=False,
    show_fig=False,
):
    test_patient_dir = img_dir / f"test_patient"
    test_patient_dir.mkdir(exist_ok=True, parents=True)

    p = T1DMPatient.withName(patient_name)
    basal_rate = p._params.u2ss * p._params.BW  # U/min

    t = []
    CHO = []
    insulin = []
    BG = []

    while p.t < 2000:
        carb = 0

        if basal_rate is not None:
            act = Action(insulin=basal_rate, CHO=carb)
        else:
            act = Action(insulin=0, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    if fig_title is None:
        if basal_rate is not None:
            fig_title = f"test_patient_{patient_name}_basal_{basal_rate:.3f}"
        else:
            fig_title = f"test_patient_{patient_name}_no_basal"

    if show_fig:
        plot_and_show(t, BG, CHO, insulin, BG[0], fig_title)
    if save_fig:
        file_name = test_patient_dir / f"{fig_title}.png"
        plot_and_save(t, BG, CHO, insulin, BG[0], file_name)


if __name__ == "__main__":
    # test patient and verify the best basal rate
    patient_group = ["adolescent", "adult", "child"]

    for patient_type in patient_group:
        for patient_id in range(1, 10):
            patient_name = f"{patient_type}#{patient_id:03d}"
            test_patient(patient_name=patient_name, save_fig=True, show_fig=False)
