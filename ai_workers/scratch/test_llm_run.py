import os
import sys
import json

from ai_workers.modules.summarization.summarizer import Summarizer

sample_utterances = [
    {"start": 0.02, "end": 20.78, "text": "One of the things that most riders have difficulty learning and Everybody wants to be able to do is to whip the bike Basically whipping the bike is exactly kind of what it sounds like you off the face of the jump Basically whip the bike out sideways so that the goal would be to get the bike either parallel to the ground"},
    {"start": 20.78, "end": 40.98, "text": "beyond which some riders are able to do on sizable jumps But really to start with it's just kind of kicking the bike out And it's a natural progression some people get confused as to what actually goes on in the whip and how you get the bike back But when you do the whip correctly the bike and the momentum of your bike will actually bring"},
    {"start": 40.98, "end": 61.70, "text": "you back around and straighten You out before you hit the ground the key thing is to not obviously go above your ability level or try to go You know too far down or just also making sure that you're on the right type of jump to be able to whip it on The key is In the approach to the jump"},
    {"start": 61.70, "end": 81.96, "text": "where you're actually almost leaning into it using the sidewalls of your tires to kind of bite as you take off the face of the jump Which will bring the bike actually flat as you leave the as your tires leave the Take off of the jump and what momentum will do with it at that point is just kind of carry it back around and then over"},
    {"start": 81.96, "end": 102.14, "text": "there's a point where if you actually still framed it where you would be flat kind of this way and then as it comes around you're gonna be out more to where your your tires or sideways from where you were taking off and that whole thing will carry through and It'll set your your bike back up straight as long as you do it right"},
    {"start": 102.58, "end": 120.74, "text": "The key is just very very small steps learning to do it kind of on On real small jumps with maybe even a hip landing to where the take off would be Straight to you but the landing or the downside where you're shooting for would be at a an angle"},
    {"start": 121.80, "end": 141.84, "text": "You know over to your left if you can whip to your left or over to your right if you're trying to do it to your right That gives you a little bit more room for a margin of error To where you don't have to bring your bike all the way back perfectly straight and you could land on the side slope and It takes a little bit of the risk out of it"},
    {"start": 178.72, "end": 198.74, "text": "As you can see the whip is basically just a combination of Taking off with the correct amount of lean and then using your body positioning To help bring the bike around Depending on how much you know how far or how how little you got the bike out The key is just"},
    {"start": 198.74, "end": 215.66, "text": "to stay within your ability level start small and work up and up and up make sure that The jump has When you're leaning against the jump your front end especially and your rear tire is not gonna break loose on the face Which would cause you to get kind of spiraling"},
    {"start": 216.38, "end": 236.80, "text": "Too far or it would make it very difficult to bring it back. So stay within your ability level Use your your body and your bike and let the momentum carry it across and you'll be whipping in no time"},
    {"start": 236.80, "end": 256.90, "text": "The whip is by far the most impressive looking trick and if you can master it you will be the envy of all your buddies The important thing as I said earlier is just to take it slow Analyze, you know the face of the jump find the right jump that you feel comfortable on Begin the lean, you know, let the bike follow through on the momentum and your body position correct your Correct your trajectory to get you back to straight"},
    {"start": 256.90, "end": 268.98, "text": "as you get this feeling and you do it repetitively You'll be able to go a little bit farther a little bit farther and and it can stay completely in control Like I said the whip is a lot of fun. It's fun to do. It's fun to watch and Everybody should go try one"}
]

sample_chapters = [
    {"title": "Giới thiệu kỹ thuật Bike Whip", "startTime": 0.0, "endTime": 40.0, "summary": "Giới thiệu"},
    {"title": "Kỹ thuật tiếp cận dốc & Quán tính", "startTime": 40.0, "endTime": 102.0, "summary": "Kỹ thuật tiếp cận dốc"},
    {"title": "Cách tập luyện an toàn trên dốc nhỏ", "startTime": 102.0, "endTime": 141.0, "summary": "Cách tập luyện an toàn"},
    {"title": "Thực hành biểu diễn kỹ thuật Whip", "startTime": 141.0, "endTime": 198.0, "summary": "Biểu diễn thực tế"},
    {"title": "Tổng kết kỹ thuật & Lời khuyên", "startTime": 198.0, "endTime": 269.0, "summary": "Tổng kết kỹ thuật"}
]

sample_slides = [
    {"start_seconds": 0.0, "end_seconds": 40.0, "start_timecode": "00:00", "end_timecode": "00:40", "caption": "Người lái xe giới thiệu về kỹ thuật whip xe cào cào", "object_counts": {"person": 1, "motorcycle": 1}},
    {"start_seconds": 40.0, "end_seconds": 102.0, "start_timecode": "00:40", "end_timecode": "01:42", "caption": "Phân tích tư thế nghiêng người và góc tiếp cận dốc nhảy", "object_counts": {"person": 1, "motorcycle": 1}},
    {"start_seconds": 102.0, "end_seconds": 141.0, "start_timecode": "01:42", "end_timecode": "02:21", "caption": "Hướng dẫn chọn dốc nhỏ và góc đáp an toàn", "object_counts": {"person": 1, "motorcycle": 1}},
    {"start_seconds": 147.0, "end_seconds": 178.0, "start_timecode": "02:27", "end_timecode": "02:58", "caption": "Pha thực hành bay người và vẩy đuôi xe đạp/mô tô trên không", "object_counts": {"person": 1, "motorcycle": 1}},
    {"start_seconds": 198.0, "end_seconds": 269.0, "start_timecode": "03:18", "end_timecode": "04:29", "caption": "Người hướng dẫn đứng cạnh xe giải thích tổng kết", "object_counts": {"person": 1, "motorcycle": 1}}
]

summarizer = Summarizer()
res = summarizer.summarize(sample_utterances, sample_slides, sample_chapters, job_id="test-live")
print("=== RESULT KEYS ===")
print("video_title:", res.get("video_title"))
print("summary:", repr(res.get("summary")))
print("chapters:", json.dumps(res.get("chapters"), ensure_ascii=False, indent=2))
print("key_takeaways:", json.dumps(res.get("key_takeaways"), ensure_ascii=False, indent=2))
