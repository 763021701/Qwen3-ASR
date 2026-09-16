import unittest

from evaluation.english_medical.eval_english_medical_asr_jsonl import normalize_english


class EnglishMedicalNormalizationTest(unittest.TestCase):
    def test_multiplication_variants_match(self):
        expected = normalize_english("5 cm x 5 cm")
        self.assertEqual(normalize_english("5 cm × 5 cm"), expected)
        self.assertEqual(normalize_english("5 cm 乘 五 cm"), expected)

    def test_chinese_numerals_match_arabic(self):
        self.assertEqual(normalize_english("一點五 cm"), normalize_english("1.5 cm"))
        self.assertEqual(normalize_english("二零零八"), normalize_english("2008"))

    def test_english_number_words_match_arabic(self):
        self.assertEqual(normalize_english("two blocks"), normalize_english("2 blocks"))
        self.assertEqual(normalize_english("three paraffin"), normalize_english("3 paraffin"))
        self.assertEqual(normalize_english("one block"), "1 block")
        self.assertEqual(normalize_english("1 block"), "1 block")

    def test_cantonese_two_matches_arabic(self):
        self.assertEqual(normalize_english("兩 cm"), normalize_english("2 cm"))
        self.assertEqual(normalize_english("兩點八 cm"), normalize_english("2.8 cm"))

    def test_spaced_units_and_acronyms_match(self):
        self.assertEqual(normalize_english("5 m m x 5 c m"), normalize_english("5 mm x 5 cm"))
        self.assertEqual(normalize_english("V A B"), normalize_english("VAB"))

    def test_spaced_sn_labels_stay_separate_arabic_numbers(self):
        hyp = "S N 一, S N 二 and non S N. S N 一"
        ref = "'SN 1', 'SN 2' and 'non-SN'. SN 1"
        expected = "sn 1 sn 2 and non sn sn 1"
        self.assertEqual(normalize_english(hyp), expected)
        self.assertEqual(normalize_english(ref), expected)
        self.assertNotIn("ones", normalize_english(hyp))
        self.assertNotIn("snsn", normalize_english(hyp))

    def test_alphanumeric_specimen_ids_match(self):
        expected = normalize_english("26SS11731")
        self.assertEqual(normalize_english("26 SS 11731"), expected)
        self.assertEqual(normalize_english("二六 S S 一一七三一"), expected)

    def test_gram_variants_match(self):
        expected = normalize_english("5 gram")
        for value in ("5 gramme", "5 gm", "5 grams"):
            self.assertEqual(normalize_english(value), expected)

    def test_paraffine_matches_paraffin(self):
        self.assertEqual(normalize_english("paraffine"), normalize_english("paraffin"))

    def test_decimal_point_is_preserved(self):
        self.assertEqual(normalize_english("6.3 cm"), "6.3 cm")
        self.assertNotEqual(normalize_english("6.3 cm"), normalize_english("63 cm"))

    def test_cantonese_classifier_is_removed_after_number(self):
        expected = normalize_english("4 blocks")
        self.assertEqual(normalize_english("4 個 blocks"), expected)
        self.assertEqual(normalize_english("4個 blocks"), expected)


    def test_known_hyphenated_compounds_match_closed_spelling(self):
        pairs = (
            ("antero-posteriorly", "anteroposteriorly"),
            ("antero-inferior", "anteroinferior"),
            ("antero-superior", "anterosuperior"),
            ("medio-laterally", "mediolaterally"),
            ("micro-nodules", "micronodules"),
            ("supero-inferiorly", "superoinferiorly"),
        )
        for hyphenated, closed in pairs:
            self.assertEqual(normalize_english(hyphenated), normalize_english(closed))

    def test_unicode_hyphen_uses_compound_dictionary(self):
        self.assertEqual(
            normalize_english("antero‑posteriorly"),
            normalize_english("anteroposteriorly"),
        )

    def test_other_hyphens_become_spaces(self):
        pairs = (
            ("non-SN", "non SN"),
            ("T-code", "T code"),
            ("well-differentiated", "well differentiated"),
            ("salpingo-oophorectomy", "salpingo oophorectomy"),
            ("2-3 cm", "2 3 cm"),
        )
        for hyphenated, spaced in pairs:
            self.assertEqual(normalize_english(hyphenated), normalize_english(spaced))

    def test_parentheses_are_removed_but_content_is_preserved(self):
        self.assertEqual(normalize_english("(A)"), "a")
        self.assertEqual(normalize_english("(F to H)"), "f to h")
        self.assertEqual(normalize_english("(係總數)"), "係總數")
        self.assertEqual(
            normalize_english("Block (A) and (F to H) (係總數)"),
            "block a and f to h 係總數",
        )

    def test_liang_variant_matches_arabic_two(self):
        self.assertEqual(normalize_english("倆個 blocks"), normalize_english("2 blocks"))
        self.assertEqual(normalize_english("倆 cm"), normalize_english("2 cm"))

    def test_nian_prefix_matches_arabic_twenty(self):
        self.assertEqual(normalize_english("廿八 cm"), normalize_english("28 cm"))
        self.assertEqual(normalize_english("廿"), normalize_english("20"))

    def test_long_spelled_letters_split_into_words(self):
        self.assertEqual(
            normalize_english("M A R K E L Y M A R K E D L Y"),
            "markely markedly",
        )

    def test_block_ranges_do_not_merge_single_letters(self):
        self.assertEqual(normalize_english("A to D A to D"), "a to d a to d")
        self.assertEqual(normalize_english("Blocks C D"), "blocks c d")

    def test_spaced_antero_posteriorly_matches_closed_form(self):
        self.assertEqual(
            normalize_english("antero posteriorly"),
            normalize_english("anteroposteriorly"),
        )
        self.assertEqual(
            normalize_english("1.8 cm antero posteriorly"),
            normalize_english("1.8 cm anteroposteriorly"),
        )

    def test_oclock_is_not_merged_with_hour_number(self):
        self.assertEqual(normalize_english("3 to 9 o'clock"), "3 to 9 oclock")
        self.assertEqual(normalize_english("12 o'clock"), "12 oclock")
        self.assertEqual(
            normalize_english("from 3 to 9 o'clock"),
            normalize_english("from 3 to 9 o clock"),
        )

    def test_repeated_two_does_not_become_twenty_two(self):
        self.assertEqual(
            normalize_english("two, two collapsed"),
            normalize_english("2 2 collapsed"),
        )

    def test_level_roman_numerals_match_arabic(self):
        self.assertEqual(normalize_english("level I lymph nodes"), "level 1 lymph nodes")
        self.assertEqual(normalize_english("level II lymph nodes"), "level 2 lymph nodes")
        self.assertEqual(
            normalize_english("level I lymph nodes"),
            normalize_english("level 1 lymph nodes"),
        )
        self.assertEqual(
            normalize_english("level II lymph nodes"),
            normalize_english("level 2 lymph nodes"),
        )


if __name__ == "__main__":
    unittest.main()

