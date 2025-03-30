import logging

from simglucose.patient.t1dpatient import T1DPatient, Action
from test_utils import plot_and_show

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


patient_name = "adolescent#003"
p = T1DPatient.withName(patient_name)
basal = p._params.u2ss * p._params.BW / 6000  # U/min


def test_patient(
    patient_name="adolescent#003", basal_rate=0.2, fig_title=None, use_basal=False
):
    p = T1DPatient.withName(patient_name)

    t = []
    CHO = []
    insulin = []
    BG = []

    while p.t < 2000:
        carb = 0

        if use_basal:
            act = Action(insulin=basal_rate, CHO=carb)
        else:
            act = Action(insulin=0, CHO=carb)

        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    if fig_title is None:
        fig_title = f"test_patient_{patient_name}"
    plot_and_show(t, BG, CHO, insulin, BG[0], fig_title)


if __name__ == "__main__":
    # test patient
    test_patient(patient_name="adolescent#003", use_basal=True, basal_rate=0.011)
    test_patient(patient_name="adolescent#001", use_basal=True, basal_rate=0.014)
