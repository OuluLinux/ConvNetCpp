import recognition


def process_frame():
    # Composite token heads (legacy key-space reused intentionally)
    for slot_id in [
        "figure_token_1", "figure_token_2", "figure_token_3", "figure_token_4", "figure_token_5",
        "header_token_1", "header_token_2",
    ]:
        token = recognition.get_label(slot_id)
        if token:
            recognition.set_meta(slot_id, token)

    # Bool gates
    for slot_id in ["figure_block", "equation_block", "caption_block"]:
        if recognition.get_bool(slot_id):
            recognition.set_meta(slot_id, "true")
