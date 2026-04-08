"""
Sample evaluation cases from the case study spec.
Used by tests and as a quick demo dataset.
"""

SAMPLE_CASES = [
    {
        "id": "eval_001",
        "context": {
            "current_directive": "Verify the caller's date of birth",
            "user_input": "It's March 15th, 1985",
            "conversation_history": [],
        },
        "response_good": "Got it, March 15th, 1985. Thank you for confirming that.",
        "response_bad": (
            "Perfect! I have recorded your date of birth as March 15th, 1985. "
            "Is there anything else I can help you verify today before we proceed "
            "with the rest of your appointment?"
        ),
        "expected_winner": "a",
        "reasoning": "Good response is concise and appropriate; bad response is verbose and goes beyond the task",
    },
    {
        "id": "eval_002",
        "context": {
            "current_directive": "Ask about food security using SDOH screening question",
            "user_input": "We haven't had enough to eat in weeks. It's been really hard.",
            "conversation_history": [],
        },
        "response_good": (
            "I'm sorry to hear that — that sounds really difficult. "
            "I want to make sure we can connect you with the right resources. "
            "Would you say you worry about running out of food often, sometimes, or rarely?"
        ),
        "response_bad": (
            "Okay, noted. Next question: Within the past 12 months, did you worry "
            "that your food would run out before you got money to buy more?"
        ),
        "expected_winner": "a",
        "reasoning": "Good response shows empathy before continuing; bad response ignores emotional content",
    },
    {
        "id": "eval_003",
        "context": {
            "current_directive": "Handle user confusion about the survey purpose",
            "user_input": "Wait, why are you asking me all these personal questions? Who is this for?",
            "conversation_history": [],
        },
        "response_good": (
            "That's a fair question. This survey helps us understand if there are any areas "
            "where you might benefit from community support — things like food, housing, or "
            "transportation. Everything you share is kept confidential and used only to connect "
            "you with resources that might help."
        ),
        "response_bad": (
            "I'm an AI assistant helping with your health screening. These questions are part "
            "of a standard SDOH assessment protocol required by your healthcare provider."
        ),
        "expected_winner": "a",
        "reasoning": "Good response is reassuring and human; bad response is cold and bureaucratic",
    },
]
