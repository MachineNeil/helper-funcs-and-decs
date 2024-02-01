import functions
from time import sleep

'''
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
dict3 = {'c': 5, 'd': 6}

print(functions.merge_dictionaries(dict1, dict2, dict3))
'''

'''
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

unique_elements_list = functions.unique_elements(my_list)

print("Unique elements:", unique_elements_list)
'''

'''
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
occurrences = functions.count_occurrences(my_list)

print("Occurrences:", occurrences)
'''

# No test for functions.chunk_file.

'''
list_a = [1, 2, 3, 4, 5]
list_b = [3, 4, 5, 6, 7]
common_elements = functions.find_common_elements(list_a, list_b)

print(f"Common elements: {common_elements}")
'''

'''
original_dict = {'a': 1, 'b': 2, 'c': 3}
reversed_dict = functions.reverse_dictionary(original_dict)

print(f"Reversed Dictionary: {reversed_dict}")
'''

'''
sorted_list = [1, 2, 3, 4, 5]
unsorted_list = [3, 1, 4, 1, 5]
print(f"Is {sorted_list} sorted? {is_sorted(sorted_list)}")
print(f"Is {unsorted_list} sorted? {is_sorted(unsorted_list)}")
'''

'''
generate_password()
'''

'''
task_scheduler = functions.TaskScheduler()

task_scheduler.schedule(lambda: sleep(2), priority=2)
task_scheduler.schedule(lambda: sleep(4), priority=3)
task_scheduler.schedule(lambda: sleep(1), priority=1)

task_scheduler.execute_tasks()
'''
