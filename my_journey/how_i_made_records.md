This is my journey on how did I made the whole system, Good for understanding thought process, Approaches and how I used AI + Other tools + Critical thinking to bring up a solution.

First i read whole problem twice/thrice and took an hour and more, Researching more on AEO and SEO before diving deep

Understanding the basic structure, now it's time to code.. before anything I used Copilot Claude 4.6 to generate use AEO module implementation (without checks).. then I ran over the code to get basic outputs like:
![alt text](some_outputs_c2.png)

Finally now time to check code and start implementing testcases for each modular checks, my plan here is to create dataset first (by reading the whole doc again, and thinking all the edge cases possible), then write it into the testcase engine and only after I am sure I covered everything will I start implementing/tweaking each checks in AEO..

CHecking at code, I noticed the testcase generated has one big file with 50+ cases written for check A, but i felt this is not clean and could be refactored- So tweaking the testcase folder structure to accurately depict each function/module and do unit testing, then on top of these unit tests, I will do one big integration test that will combine each smaller functions..

for here example we have aeo/check_a/test_has_hedge_phrase.py which just checks for hedge phrase

Now I went back to the cases and worked on each checks Step by step- First I noticed there is no check for extracting first paragraph, so I made it and then wrote some edge cases etc to verify if paragraph extract is okay or not- However in testcase and code I noticed how it was skipping paragraph which words less than 3- This is something AI Automatically generated, but since I knew we are deliberately scoring on AEO- We should not skip this and instead take small paragraph only (so we will later highlight / flag this saying this is Poor AEO optimized), therefore I removed the logic where the code skipped over small paragraph

I also deliberately removed more cases like skipping <p> tags inside headers, this is to keep the assignment simple- WHile also writing edge cases there were some instances where i had minor doubts: for example what does: Example of first paragraph break in plain text mean? so i used chatgpt to ask with some examples and aligned test case behaviour accordingly

Similiarly checking into other test cases, words and declarative felt good enough however hedge phrases had few words and I researched what all others can be said / used as hedge phrases / words, So i simply expanded the list to contain more such words

Similiarly I finished other testcases / individual tests, tweking anything required and each folder like aeo/check_... will have a file with name contianing full_integration, it will contain full data and tests for that case <read one of that file and add small example quick here>

