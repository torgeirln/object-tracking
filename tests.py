
from SOT.Sensors.sensors_tests import test_1D_detector_w_uniform_clutter, test_2D_detector_w_uniform_clutter

class SOTtests():
    @staticmethod
    def test_sensors_1D_detector_w_uniform_clutter():
        test_1D_detector_w_uniform_clutter()

    @staticmethod
    def test_sensors_2D_detector_w_uniform_clutter():
        test_2D_detector_w_uniform_clutter()


if __name__ == "__main__":
    
    # SOTtests.test_sensors_1D_detector_w_uniform_clutter()
    SOTtests.test_sensors_2D_detector_w_uniform_clutter()
