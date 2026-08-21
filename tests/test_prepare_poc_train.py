import unittest

from tools.prepare_poc_train import contains_cjk_ideograph, is_from_denoised


class PreparePocTrainTest(unittest.TestCase):
    def test_contains_cjk_ideograph(self):
        self.assertFalse(contains_cjk_ideograph("Specimen measured 5 cm × 3.5 cm."))
        self.assertTrue(contains_cjk_ideograph("第1隻 Specimen labelled left pelvic lymph nodes."))
        self.assertTrue(contains_cjk_ideograph("\U00020000"))

    def test_is_from_denoised(self):
        self.assertTrue(is_from_denoised("/data/processed/segmented/from_denoised/sample.wav"))
        self.assertFalse(is_from_denoised("/data/processed/segmented/from_raw/sample.wav"))


if __name__ == "__main__":
    unittest.main()
