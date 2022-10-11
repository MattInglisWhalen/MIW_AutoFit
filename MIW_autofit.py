
import autofit.examples.frontend_startup as fe_start

def start_autofit():
    fe_start.start_frontend()


if __name__ == "__main__":

    # start_autofit()

    possible_list1 = [1,2]
    possible_list2 = 3

    final = []

    checkers = [possible_list1,possible_list2]
    for thing in checkers :
        if isinstance(thing, list):
            final.extend(thing)
        else :
            final.append(thing)


    print(final)