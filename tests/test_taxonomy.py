from novavision import taxonomy as tx


def test_ekman_mapping_targets_are_valid():
    assert set(tx.GOEMOTIONS_TO_EKMAN.values()) <= set(tx.EMOTIONS)


def test_to_ekman_examples():
    assert tx.to_ekman("love") == "joy"
    assert tx.to_ekman("annoyance") == "anger"
    assert tx.to_ekman("grief") == "sadness"
    assert tx.to_ekman("curiosity") == "surprise"
    assert tx.to_ekman("not-a-label") == "neutral"


def test_priors_have_expected_signs():
    assert tx.prior("joy")[0] > 0
    assert tx.prior("sadness")[0] < 0
    assert tx.prior("anger")[1] > tx.prior("sadness")[1]


def test_emotion_prompts_cover_all_emotions():
    assert set(tx.EMOTION_PROMPTS) == set(tx.EMOTIONS)
