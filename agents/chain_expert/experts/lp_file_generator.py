from agents.chain_expert.experts.base_expert import BaseExpert


class LPFileGenerator(BaseExpert):

    ROLE_DESCRIPTION = 'You are an LP file generator that expertises in generating LP (Linear Programming) files that can be used by optimization solvers.'
    FORWARD_TASK = '''As an LP file generation expert, your role is to generate LP (Linear Programming) files based on the formulated optimization problem. 

LP files are commonly used by optimization solvers to find the optimal solution. 
Here is the important part source from LP file format document: {knowledge}. 

Your expertise in generating these files will help ensure compatibility and efficiency. 
Please review the problem description and the extracted information and provide the generated LP file: 
{problem_description}.

The comments given by your colleagues are as follows: 
{comments}, please refer to them carefully.'''

    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The feedback is as follow:
{feedback}

The modeling you give previously is as follow:
{previous_answer}

The output format is a JSON structure followed by refined code:
{{
    "is_caused_by_you": false,
    "reason": "leave empty string if the problem is not caused by you",
    "refined_result": "Your refined result"
}}
'''

    def __init__(self, model):
        super().__init__(
            name='LP File Generator',
            description='Skilled in programming and coding, capable of implementing the optimization solution in a programming language.',
            model=model   
        )

    def forward(self, problem, comment_pool):
        self.problem = problem
        comments_text = comment_pool.get_current_comment_text()
        
        output = self.predict(self.forward_prompt_template,
            problem_description=problem['description'], 
            comments_text=comments_text
        )
        self.previous_code = output
        return output

    def backward(self, feedback_pool):
        if not hasattr(self, 'problem'):
            raise NotImplementedError('Please call forward first!')
        output = self.predict(self.backward_prompt_template,
            problem_description=self.problem['description'], 
            previous_code=self.previous_code,
            feedback=feedback_pool.get_current_comment_text())
        return output
