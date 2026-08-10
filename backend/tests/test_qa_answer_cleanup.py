import unittest

from app.api.v1.qa import clean_qa_answer


class QAAnswerCleanupTests(unittest.TestCase):
    def test_removes_leaked_language_policy_sentence(self):
        answer = (
            "Đoạn video liên quan đến giấc ngủ và việc ghi nhớ nằm quanh [02:36].\n"
            "Vì câu hỏi yêu cầu ngôn ngữ Việt, câu trả lời được cung cấp bằng tiếng Việt tự nhiên."
        )

        cleaned = clean_qa_answer(answer)

        self.assertEqual(cleaned, "Đoạn video liên quan đến giấc ngủ và việc ghi nhớ nằm quanh [02:36].")

    def test_keeps_normal_answer_text(self):
        answer = "Video nói về nhịp sinh học và tác động của thiếu ngủ. [00:53]"

        self.assertEqual(clean_qa_answer(answer), answer)

    def test_removes_unrelated_script_characters(self):
        answer = "Hệ quả dài hạn của mất ngủ对健康 và hiệu suất học tập."

        self.assertEqual(clean_qa_answer(answer), "Hệ quả dài hạn của mất ngủ và hiệu suất học tập.")

    def test_normalizes_bold_bullet_marker(self):
        answer = "**•**\nẢnh hưởng của adenosine và caffeine."

        self.assertEqual(clean_qa_answer(answer), "- Ảnh hưởng của adenosine và caffeine.")


    def test_normalizes_standalone_bullet_marker(self):
        answer = "•\nChặn adenosine và giúp tỉnh táo tạm thời."

        self.assertEqual(clean_qa_answer(answer), "- Chặn adenosine và giúp tỉnh táo tạm thời.")

    def test_normalizes_timestamp_line(self):
        answer = "*Timestamp:* ▶ [01:27] – Caffeine blocks adenosine receptors."

        self.assertEqual(clean_qa_answer(answer), "- [01:27] Caffeine blocks adenosine receptors.")


if __name__ == "__main__":
    unittest.main()
