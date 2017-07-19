# Tweet Classification for Disaster Relief

This is the homepage for the [SAILORS 2017](http://sailors.stanford.edu/) NLP research project. 
Here you can find links to all class materials used for the research project.

Instructors: [Abi See](http://cs.stanford.edu/people/abisee/) (abisee@stanford.edu), [Sebastian Schuster](http://sebschu.com/) (sebschu@stanford.edu)

## Slides
* [Day 1: Introduction to NLP](https://docs.google.com/presentation/d/1ETVn3Zpqy7Cmv7LuqG-kJ2zwR7Yz8BQUMgnzTkC6fwY/edit?usp=sharing)
* [Day 2: Rule-based classifiers](https://docs.google.com/presentation/d/1aWweVcIf1XBHu8okf5k3d_KXqObxHHx9W_ljCdm2jyw/edit?usp=sharing)
* [Day 3: Evaluation metrics](https://docs.google.com/presentation/d/1nfideUIXIcslK9eLNrtxVGyaJG8KJlq1heI-A4YJn_w/edit?usp=sharing) (Exercise sheet [here](https://docs.google.com/document/d/1IyynNr2hVJY8LOzFEBKRXNJ71usqfPQuR81lrFjEcPc/edit?usp=sharing))
* [Day 4: Probability theory and Bayes rule](https://docs.google.com/presentation/d/1nOEo5AKcdwBjdhNLj1yP0CTw508H6I34BjU6yxRZ9ck/view) 
(Exercise sheet [here](https://docs.google.com/document/d/1u8pY6YicTEa3xZI6QxcPfrZ8A9mIJYxjA4iL6hpSB9c/edit?usp=sharing))
* [Day 5 morning: Naive Bayes classifier](https://docs.google.com/presentation/d/1EM6BmNpjo5QdymzkUUfWdwNxuIw-AZtzGjswcW-ioMY/edit?usp=sharing)
* [Day 5 afternoon: More NLP](https://docs.google.com/presentation/d/1biWvkfLANZWiwvePX7WWcKI5JNNMJ9XSewZDwyHmCC8/edit?usp=sharing)
* [Day 6 morning: Naive Bayes classifier for Twitter project](https://docs.google.com/presentation/d/1qIdeh8nYIOHztvkK5DfN-86VX9RHJiV7CAOKf7MbL8M/edit?usp=sharing)
* [Day 6 afternoon: Neural Networks](https://docs.google.com/presentation/d/1D23sp1JVwPbORXlnQ8D7fpePdxjNSuoGO0IfZLKT4zY/edit?usp=sharing)
* [Day 7: Wrap-up](https://docs.google.com/presentation/d/128LUceybBh4ORH04Wwe9MqFCWz-fDJVZ08CF1q9V9Ro/edit?usp=sharing)


## Other materials
* [Day 1: Data exploration spreadsheet](https://docs.google.com/spreadsheets/d/1EC83i5jhi5TjQTT4XN0v4CScZcie9WloASPGSEdJ2mY/edit?usp=sharing)
* [Naive Bayes cheat sheet](https://docs.google.com/document/d/1Z6WnbCQYtOsaoFAZc4VdXtCc9edGIlPBX9CulSwBVgo/edit)
* [Next Steps: Resources for after SAILORS](https://docs.google.com/document/d/1_byDijN6Mc0Gk7phL5e5dmVuhyMkkZDNoEsXXvnfzPw/edit?usp=sharing)


## Instructions for running the notebooks

1. Install Anaconda.
    
    Anaconda is a python distribution that makes it really easy to install additional python packages and manage different Python versions. You can download Anaconda from https://www.continuum.io/. Make sure to download the Python 2.7 version! This should also automatically install Jupyter notebook, which you'll need to run the notebooks.

2. Install numpy and nltk:
    
    Open a Terminal window and type 
    
    ```
    conda install nltk numpy pandas
    ```

3. Copy ("clone") the GitHub repository to your computer:

    Open a Terminal window and type 
    
    ```
    git clone https://github.com/abisee/sailors2017
    ```
    
    This will copy all the notebooks to your computer.

4. Change into the directory:

   In the same Terminal window, type

   ```
   cd sailors2017
   ```

5. Download the tokenizer models:

    Start a Python console by typing `python` in the Terminal window. Then run the following commands:

    ```
    import nltk
    nltk.download("punkt")
    exit()
    ```

6. Run the jupyter notebook:

    ```
    jupyter notebook
    ```
