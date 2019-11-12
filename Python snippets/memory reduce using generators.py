# def generate_numbers():
#     n = 0
#     while n < 3:
#         yield n
#         n += 1
#
# numbers = generate_numbers()
# print(next(numbers))
# print(next(numbers))
# print(next(numbers))
# print(next(numbers))
#
# print(type(numbers))


import memory_profiler
import time

def check_even_normal(numbers):
    even = []
    for num in numbers:
        if num % 2 == 2:
            even.append(num * num)
    return even

def check_even_generator(numbers):
    for num in numbers:
        if num % 2 == 0:
            yield num * num

if __name__ == '__main__':

    m1 = memory_profiler.memory_usage()
    t1 = time.process_time()
    cubes = check_even_normal(range(100000000))
    t2 = time.process_time()
    m2 = memory_profiler.memory_usage()
    time_diff = t2 - t1
    mem_diff = m2[0] - m1[0]
    print(f"It took {time_diff} secs and {mem_diff} Mb to execute normal method!")

    m1 = memory_profiler.memory_usage()
    t1 = time.process_time()
    cubes = check_even_generator(range(100000000))
    t2 = time.process_time()
    m2 = memory_profiler.memory_usage()
    time_diff = t2 - t1
    mem_diff = m2[0] - m1[0]
    print(f"It took {time_diff} secs and {mem_diff} Mb to execute generator method!")
