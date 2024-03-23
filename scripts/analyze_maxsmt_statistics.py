import json
import operator
from functools import reduce
from pathlib import Path
from sys import argv


def main(args):
    analyze_maxsmt_statistics(Path(args[1]), Path(args[2]))


def analyze_maxsmt_statistics(stat_file, analyzed_stat_file_to_save):
    if not stat_file.exists() or not stat_file.is_file():
        raise FileExistsError(f"File with statistics [{str(stat_file)}] does not exist")

    with open(stat_file, "r", encoding="utf-8") as f:
        stat = json.load(f)

    logics_size = len(stat["logics"])
    logics_statistics = []

    def obj_dict(obj):
        return obj.__dict__

    for i in range(0, logics_size):
        logic_stat = create_logic_statistics((stat["logics"])[i])
        logics_statistics.append(logic_stat)
        logic_stat_str = json.dumps(logic_stat, default=obj_dict, indent=2, separators=(',', ': '))
        print(logic_stat_str + "\n")

    with open(analyzed_stat_file_to_save, "w", encoding="utf-8") as f:
        json.dump(logics_statistics, f, default=obj_dict, indent=2, separators=(',', ': '))


def create_tests_size_statistics(tests):
    tests_size = len(tests)
    tests_executed_maxsmt_size = len(list(filter(lambda x: x.get("maxSMTCallStatistics") is not None, tests)))
    tests_executed_maxsmt_passed_tests_percent = 0 if tests_executed_maxsmt_size == 0 \
        else len(list(filter(lambda x: x["passed"], tests))) / tests_executed_maxsmt_size * 100
    failed_or_ignored_tests_size = len(list(filter(lambda x: not x["passed"], tests)))
    failed_tests_wrong_soft_constr_sum_size = len(list(filter(lambda x: x["checkedSoftConstraintsSumIsWrong"], tests)))
    ignored_tests_size = len(list(filter(lambda x: x["ignoredTest"], tests)))
    failed_on_parsing_or_converting_expressions_size = len(
        list(filter(lambda x: x["failedOnParsingOrConvertingExpressions"], tests)))

    def get_unique_exception_messages(collection):
        # Set is converted to list in order to dump statistics to JSON (otherwise, the script fails
        # with such error: 'AttributeError: 'set' object has no attribute '__dict__'.').
        return list(
            set(map(lambda x: None if x.get("exceptionMessage") is None else x["exceptionMessage"], collection)))

    failed_on_parsing_or_converting_expressions_exception_messages = get_unique_exception_messages(
        list(filter(lambda x: x["failedOnParsingOrConvertingExpressions"], tests)))

    return TestsSizeStatistics(tests_size, tests_executed_maxsmt_size,
                               tests_executed_maxsmt_passed_tests_percent,
                               failed_or_ignored_tests_size, ignored_tests_size,
                               failed_on_parsing_or_converting_expressions_size,
                               failed_on_parsing_or_converting_expressions_exception_messages,
                               failed_tests_wrong_soft_constr_sum_size)


def create_tests_queries_to_solver_statistics(tests):
    def max_smt_stat(test):
        return test["maxSMTCallStatistics"]

    def queries_to_solver_number(test):
        if isinstance(test, int):
            return test
        return max_smt_stat(test)["queriesToSolverNumber"]

    def time_in_solver_queries_ms(test):
        if isinstance(test, int):
            return test
        return max_smt_stat(test)["timeInSolverQueriesMs"]

    def elapsed_time_ms(test):
        return max_smt_stat(test)["elapsedTimeMs"]

    tests_executed_maxsmt = list(filter(lambda x: x.get("maxSMTCallStatistics") is not None, tests))
    tests_executed_maxsmt_size = len(tests_executed_maxsmt)

    tests_executed_maxsmt_passed = list(filter(lambda x: x["passed"], tests_executed_maxsmt))
    tests_executed_maxsmt_passed_size = len(tests_executed_maxsmt_passed)

    tests_executed_maxsmt_failed = list(filter(lambda x: not x["passed"], tests_executed_maxsmt))
    tests_executed_maxsmt_failed_size = len(tests_executed_maxsmt_failed)

    avg_queries_to_solver_number = 0 if tests_executed_maxsmt_size == 0 else reduce(
        lambda x, y: queries_to_solver_number(x) + queries_to_solver_number(y),
        tests_executed_maxsmt, 0) / tests_executed_maxsmt_size

    avg_queries_to_solver_passed_tests_number = 0 if tests_executed_maxsmt_passed_size == 0 else reduce(
        lambda x, y: queries_to_solver_number(x) + queries_to_solver_number(y),
        tests_executed_maxsmt_passed, 0) / tests_executed_maxsmt_passed_size

    avg_queries_to_solver_failed_tests_number = 0 if tests_executed_maxsmt_failed_size == 0 else reduce(
        lambda x, y: queries_to_solver_number(x) + queries_to_solver_number(y),
        tests_executed_maxsmt_failed, 0) / tests_executed_maxsmt_failed_size

    def is_zero(value):
        return abs(value) < 0.00000001

    def avg_time_per_solver_queries_percent_list(bunch_of_tests):
        return map(lambda x: time_in_solver_queries_ms(x) / elapsed_time_ms(x) * 100 if not is_zero(
            elapsed_time_ms(x)) else elapsed_time_ms(x), bunch_of_tests)

    avg_time_per_solver_queries_percent = \
        0 if tests_executed_maxsmt_size == 0 else reduce(operator.add, avg_time_per_solver_queries_percent_list(
            tests_executed_maxsmt),
                                                         0) / tests_executed_maxsmt_size
    avg_time_per_solver_queries_passed_tests_percent = \
        0 if tests_executed_maxsmt_passed_size == 0 else reduce(operator.add, avg_time_per_solver_queries_percent_list(
            tests_executed_maxsmt_passed),
                                                                0) / tests_executed_maxsmt_passed_size

    avg_time_per_solver_queries_failed_tests_percent = \
        0 if tests_executed_maxsmt_failed_size == 0 else reduce(operator.add, avg_time_per_solver_queries_percent_list(
            tests_executed_maxsmt_failed),
                                                                0) / tests_executed_maxsmt_failed_size

    return TestsQueriesToSolverStatistics(avg_queries_to_solver_number, avg_queries_to_solver_passed_tests_number,
                                          avg_queries_to_solver_failed_tests_number,
                                          avg_time_per_solver_queries_percent,
                                          avg_time_per_solver_queries_passed_tests_percent,
                                          avg_time_per_solver_queries_failed_tests_percent)


def create_tests_elapsed_time_statistics(tests):
    def max_smt_stat(test):
        return test["maxSMTCallStatistics"]

    def elapsed_time_ms(test):
        if isinstance(test, int):
            return test
        return max_smt_stat(test)["elapsedTimeMs"]

    tests_executed_maxsmt = list(filter(lambda x: x.get("maxSMTCallStatistics") is not None, tests))
    tests_executed_maxsmt_size = len(tests_executed_maxsmt)

    avg_elapsed_time_ms = 0 if tests_executed_maxsmt_size == 0 else reduce(
        lambda x, y: elapsed_time_ms(x) + elapsed_time_ms(y), tests_executed_maxsmt,
        0) / tests_executed_maxsmt_size

    passed_tests = list(filter(lambda x: x["passed"], tests_executed_maxsmt))
    avg_elapsed_passed_tests_time_ms = 0 if tests_executed_maxsmt_size == 0 else reduce(
        lambda x, y: elapsed_time_ms(x) + elapsed_time_ms(y), passed_tests,
        0) / tests_executed_maxsmt_size

    failed_tests = list(filter(lambda x: not x["passed"], tests_executed_maxsmt))
    avg_elapsed_failed_tests_time_ms = 0 if tests_executed_maxsmt_size == 0 else reduce(
        lambda x, y: elapsed_time_ms(x) + elapsed_time_ms(y), failed_tests,
        0) / tests_executed_maxsmt_size

    return TestsElapsedTimeStatistics(avg_elapsed_time_ms, avg_elapsed_passed_tests_time_ms,
                                      avg_elapsed_failed_tests_time_ms)


class MaxSMTContext:
    def __int__(self, strategy, prefer_large_weight_constraints_for_cores, minimize_cores, get_multiple_cores):
        self.strategy = strategy
        self.prefer_large_weight_constraints_for_cores = prefer_large_weight_constraints_for_cores
        self.minimize_cores = minimize_cores
        self.get_multiple_cores = get_multiple_cores


class TestsSizeStatistics:
    def __init__(self, tests_size, tests_executed_maxsmt_size,
                 tests_executed_maxsmt_passed_tests_percent,
                 failed_tests_size,
                 ignored_tests_size,
                 failed_on_parsing_or_converting_expressions_size,
                 failed_on_parsing_or_converting_expressions_exception_messages,
                 failed_tests_wrong_soft_constr_sum_size):
        self.tests_size = tests_size
        self.tests_executed_maxsmt_size = tests_executed_maxsmt_size
        self.tests_executed_maxsmt_passed_tests_percent = tests_executed_maxsmt_passed_tests_percent,
        self.failed_tests_size = failed_tests_size
        self.ignored_tests_size = ignored_tests_size
        self.failed_on_parsing_or_converting_expressions_size = failed_on_parsing_or_converting_expressions_size
        self.failed_on_parsing_or_converting_expressions_exception_messages = (
            failed_on_parsing_or_converting_expressions_exception_messages)
        self.failed_tests_wrong_soft_constr_sum_size = failed_tests_wrong_soft_constr_sum_size


class TestsQueriesToSolverStatistics:
    def __init__(self, avg_queries_to_solver_number, avg_queries_to_solver_passed_tests_number,
                 avg_queries_to_solver_failed_tests_number, avg_time_per_solver_queries_percent,
                 avg_time_per_solver_queries_passed_tests_percent, avg_time_per_solver_queries_failed_tests_percent):
        self.avg_queries_to_solver_number = avg_queries_to_solver_number
        self.avg_queries_to_solver_passed_tests_number = avg_queries_to_solver_passed_tests_number
        self.avg_queries_to_solver_failed_tests_number = avg_queries_to_solver_failed_tests_number
        self.avg_time_per_solver_queries_percent = avg_time_per_solver_queries_percent
        self.avg_time_per_solver_queries_passed_tests_percent = avg_time_per_solver_queries_passed_tests_percent
        self.avg_time_per_solver_queries_failed_tests_percent = avg_time_per_solver_queries_failed_tests_percent


class TestsElapsedTimeStatistics:
    def __init__(self, avg_elapsed_time_ms, avg_elapsed_passed_tests_time_ms,
                 avg_elapsed_failed_tests_time_ms):
        self.avg_elapsed_time_ms = avg_elapsed_time_ms
        self.avg_elapsed_passed_tests_time_ms = avg_elapsed_passed_tests_time_ms
        self.avg_elapsed_failed_tests_time_ms = avg_elapsed_failed_tests_time_ms


class LogicTestsStatistics:
    def __init__(self, smt_solver, name, timeout_ms, max_smt_ctx, tests_size_stat: TestsSizeStatistics,
                 tests_queries_to_solver_stat: TestsQueriesToSolverStatistics,
                 tests_elapsed_time_stat: TestsElapsedTimeStatistics):
        self.smt_solver = smt_solver
        self.name = name
        self.timeout_ms = timeout_ms
        self.max_smt_ctx = max_smt_ctx
        self.tests_size_stat = tests_size_stat
        self.tests_queries_to_solver_stat = tests_queries_to_solver_stat
        self.tests_elapsed_time_stat = tests_elapsed_time_stat


def create_logic_statistics(logic):
    tests = logic["TESTS"]
    first_test = tests[0]
    test_executed_maxsmt = filter(lambda x: x.get("maxSMTCallStatistics") is not None, tests)
    first_max_smt_call_stat = None if test_executed_maxsmt is None else (list(test_executed_maxsmt)[0])[
        "maxSMTCallStatistics"]
    return LogicTestsStatistics(first_test["smtSolver"], logic["NAME"], first_max_smt_call_stat["timeoutMs"],
                                first_max_smt_call_stat["maxSmtCtx"], create_tests_size_statistics(tests),
                                create_tests_queries_to_solver_statistics(tests),
                                create_tests_elapsed_time_statistics(tests))


if __name__ == "__main__":
    main(argv)
