import unittest

from qwen_asr.inference.utils import (
    normalize_language_spec,
    parse_asr_output,
    validate_language_spec,
)


class LanguageSpecTest(unittest.TestCase):
    def test_normalize_language_spec(self):
        self.assertEqual(normalize_language_spec("chinese, english"), "Chinese,English")
        self.assertEqual(normalize_language_spec("English,Chinese"), "Chinese,English")

    def test_parse_asr_output_reorders_meta_to_supported_list_order(self):
        self.assertEqual(
            parse_asr_output("language English,Chinese<asr_text>hello 你好"),
            ("Chinese,English", "hello 你好"),
        )
        validate_language_spec("Chinese,English")

    def test_parse_asr_output_codeswitch_meta(self):
        self.assertEqual(
            parse_asr_output("language Chinese,English<asr_text>hello 你好"),
            ("Chinese,English", "hello 你好"),
        )

    def test_parse_asr_output_user_language_spec(self):
        self.assertEqual(
            parse_asr_output("raw text", user_language="chinese, english"),
            ("Chinese,English", "raw text"),
        )


if __name__ == "__main__":
    unittest.main()
