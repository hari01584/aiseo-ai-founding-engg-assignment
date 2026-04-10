This is my journey on how did I made the whole system, Good for understanding thought process, Approaches and how I used AI + Other tools + Critical thinking to bring up a solution.

First i read whole problem twice/thrice and took an hour and more, Researching more on AEO and SEO before diving deep

Understanding the basic structure, now it's time to code.. before anything I used Copilot Claude 4.6 to generate use AEO module implementation (without checks).. then I ran over the code to get basic outputs like:
![alt text](some_outputs_c2.png)

Finally now time to check code and start implementing testcases for each modular checks, my plan here is to create dataset first (by reading the whole doc again, and thinking all the edge cases possible), then write it into the testcase engine and only after I am sure I covered everything will I start implementing/tweaking each checks in AEO..

CHecking at code, I noticed the testcase generated has one big file with 50+ cases written for check A, but i felt this is not clean and could be refactored- So tweaking the testcase folder structure to accurately depict each function/module and do unit testing, then on top of these unit tests, I will do one big integration test that will combine each smaller functions..

for here example we have aeo/check_a/test_has_hedge_phrase.py which just checks for hedge phrase