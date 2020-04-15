# encoding: utf-8
"""
@author: guanshun
@contact: yashun.zys@alibaba-inc.com
@time: 2019-07-02 15:35
@file: runMain.py.py
@desc:
"""

# -*- coding: UTF-8 -*-
import unittest

from xmlrunner import xmlrunner

if __name__ == '__main__':
    suite = unittest.TestSuite()
    # 找到目录下所有的以_test结尾的py文件
    all_cases = unittest.defaultTestLoader.discover('.', '*_test.py')
    for case in all_cases:
        print(case)
        # 把所有的测试用例添加进来
        suite.addTests(case)

    runner = xmlrunner.XMLTestRunner(output='report')
    runner.run(suite)
