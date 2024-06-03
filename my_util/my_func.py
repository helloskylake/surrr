# to_check:list target:str
def check_collision(to_check, target, all_contact_pairs):
    for check in to_check:
        if (check, target) in all_contact_pairs:
            return True
    return False
