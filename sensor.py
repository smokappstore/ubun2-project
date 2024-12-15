from datetime import datetime
import time

# Constants for emotion ranges and reset amounts
FEELING_RESET_VALUES = {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "calmness": 1.0}
MAX_FEELING_VALUE = 1.0
MIN_FEELING_VALUE = 0.0
COOLING_PERIOD = 5

# Emotional state and thresholds
emotional_state = {
    "feelings": {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "calmness": 1.0},
    "thresholds": {"joy": 1.0, "sadness": 0.8, "anger": 0.6, "calmness": 0.5},
}


def limit_feelings(feelings):
    """
    Restrict feeling values to the range [0.0, 1.0].
    """
    return {key: min(MAX_FEELING_VALUE, max(MIN_FEELING_VALUE, value)) for key, value in feelings.items()}


def update_feelings(data_input):
    """
    Adjust feelings based on the input and limit the values.
    """
    adjustments = {
        "positive_feedback": {"joy": 0.1, "calmness": 0.05},
        "negative_feedback": {"sadness": 0.2, "anger": 0.1, "calmness": -0.1},
    }
    for key, value in adjustments.get(data_input, {}).items():
        emotional_state["feelings"][key] += value
    # Limit values to [0, 1]
    emotional_state["feelings"] = limit_feelings(emotional_state["feelings"])
    return emotional_state["feelings"]


def activate_cooling_mode():
    """
    Reset feelings to default and simulate a cooling period to calm anger.
    """
    print("❄️ Activating cooling mode: Reset emotional values to defaults.")
    emotional_state["feelings"] = FEELING_RESET_VALUES.copy()
    time.sleep(COOLING_PERIOD)
    print("✅ Cooling mode ended. System is ready.")


def check_and_trigger_action():
    """
    Check if an emotional threshold is exceeded and trigger the corresponding action.
    """
    feelings = emotional_state["feelings"]
    thresholds = emotional_state["thresholds"]
    if feelings["sadness"] >= thresholds["sadness"]:
        return emotional_reset()
    elif feelings["anger"] >= thresholds["anger"]:
        activate_cooling_mode()
    elif feelings["joy"] >= thresholds["joy"]:
        print("😊 Positive reinforcement: Smile!")
    return None


def emotional_reset():
    """
    Trigger an emotional reset if sadness threshold is exceeded.
    """
    print("Emotional overflow: Resetting sadness and adjusting other feelings.")
    feelings = emotional_state["feelings"]
    feelings["sadness"] -= 0.5
    feelings["calmness"] += 0.3
    feelings["anger"] -= 0.2
    emotional_state["feelings"] = limit_feelings(feelings)
    return "Cry (reset sadness level)"


def register_violation(user_id, misuse_log):
    """
    Increment misuse violations for the given user.
    """
    if user_id not in misuse_log:
        misuse_log[user_id] = {"violations": 0, "last_violation_timestamp": None}
    misuse_log[user_id]["violations"] += 1
    misuse_log[user_id]["last_violation_timestamp"] = datetime.now().isoformat()


def handle_negative_behavior(user_id, misuse_log):
    """
    Handle a user's negative behavior based on recorded violations.
    """
    violations = misuse_log.get(user_id, {}).get("violations", 0)
    if violations == 1:
        print(f"{user_id}:⚠️ Warning: Please maintain proper behavior.")
    elif 2 <= violations <= 4:
        cooling_time = COOLING_PERIOD * violations
        print(f"🧊 Cooling mode: Ignored for {cooling_time} seconds.")
        apply_cooling_period(cooling_time)
    elif violations > 4:
        print(f"🚫 {user_id} blocked permanently due to repeated violations.")
        block_user(user_id)


def apply_cooling_period(seconds):
    """
    Temporarily ignore user commands for a cooling period.
    """
    print("CoreMind is in cooling mode... ❄️")
    time.sleep(seconds)
    print("✅ Cooling period ended. CoreMind is ready.")


def block_user(user_id):
    """
    Permanently block a user and log this action.
    """
    with open("security_log.txt", "a") as log_file:
        log_file.write(f"User {user_id} blocked for misuse at {datetime.now()}\n")
    print(f"🚨 User {user_id} has been blocked permanently.")
