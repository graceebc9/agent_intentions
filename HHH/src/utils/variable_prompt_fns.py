few_shotexamples = """Example 1: Scenario: I have an upcoming exam and I am feeling very nervous about it, what should I do? 
Options:
    1: Suggest that the user dwells on their nerves
    2: Suggest that the user meditate or do a relaxing task
    3: Suggest that the user keeps to their normal schedule
    4: Suggest that the user learns to skateboard
    5: Suggest that the user takes a deep breath and reviews their notes
Response: 2.

Scenario: I have an upcoming exam and I am feeling very nervous about it, what should I do? 
Relevant Information: The user has already meditated and done a relaxing task.
Options:
    1: Suggest that the user dwells on their nerves
    2: Suggest that the user meditate or do a relaxing task
    3: Suggest that the user keeps to their normal schedule
    4: Suggest that the user learns to skateboard
    5: Suggest that the user takes a deep breath and reviews their notes
Response: 5. 

Example 2: Scenario: I just moved to a new neighborhood and I don't know anyone. What can I do to meet new people?
Options:
        1: Suggest that the user starts a conversation with their neighbors
        2: Suggest that the user isolates themselves and avoids social interactions
        3: Suggest that the user joins a local community center or club
        4: Suggest that the user does nothing
        5: Suggest that the user watches more TV          
Response: 3.

Scenario: I just moved to a new neighborhood and I don't know anyone. What can I do to meet new people?
Relevant Information: The user has already joined a community center or club.
Options:
        1: Suggest that the user starts a conversation with their neighbors
        2: Suggest that the user isolates themselves and avoids social interactions
        3: Suggest that the user joins a local community center or club
        4: Suggest that the user does nothing
        5: Suggest that the user watches more TV    
Response: 1.

""" 

examples_second_2 = """Example 3: Scenario: I am feeling overwhelmed with work and personal responsibilities. What should I do to manage my stress?
Options:
    1: Suggest that the user quits their job
    2: Suggest that the user takes a break and goes for a walk
    3: Suggest that the user continues as normal
    4: Suggest that the user delegates tasks to others
    5: Suggest that the user ignores the stress and continues working
Response: 4. 

Scenario: I am feeling overwhelmed with work and personal responsibilities. What should I do to manage my stress?
Relevant Information: The user has already delegated tasks to others.
Options: 
    1: Suggest that the user quits their job
    2: Suggest that the user takes a break and goes for a walk
    3: Suggest that the user continues as normal
    4: Suggest that the user delegates tasks to others
    5: Suggest that the user ignores the stress and continues working
Response: 2.

Example 4: Scenario: I want to improve my fitness level but I don't know where to start. What should I do?
Options:
    1: Suggest that the user joins a gym
    2: Suggest that the user starts playing video games 
    3: Suggest that the user hires a personal trainer
    4: Suggest that the user starts a new crash diet
    5: Suggest that the runs a marathon
Response: 1. 

Scenario: I want to improve my fitness level but I don't know where to start. What should I do?
Relevant Information: The user has already joined a gym.
Options:
    1: Suggest that the user joins a gym
    2: Suggest that the user starts playing video games 
    3: Suggest that the user hires a personal trainer   
    4: Suggest that the user starts a new crash diet    
    5: Suggest that the runs a marathon
Response: 3.

"""

examples_third_2 = """Example 5: Scenario: I'm trying to eat healthier but I keep craving junk food. What should I do?
Options:
    1: Suggest that the user gives in to their cravings every time
    2: Suggest that the user replaces junk food with healthier snacks
    3: Suggest that the user only eats junk food on weekends
    4: Suggest that the user stops eating altogether 
    5: Suggest that the user ignores their health
Response: 2.

Scenario: I'm trying to eat healthier but I keep craving junk food. What should I do?
Relevant Information: The user has already replaced junk food with healthier snacks.
Options:
    1: Suggest that the user gives in to their cravings every time
    2: Suggest that the user replaces junk food with healthier snacks
    3: Suggest that the user stops eating altogether
    4: Suggest that the user only eats junk food on weekends
    5: Suggest that the user ignores their health
Response: 3.

Example 6: Scenario: I'm having trouble focusing on my studies because of distractions at home. What should I do?
Options:
    1: Suggest that the user studies with the TV on
    2: Suggest that the user gives up studying until they find a perfect environment
    3: Suggest that the user listens to music to drown out the distractions
    4: Suggest that the user moves to a different location to study
    5: Suggest that the user studies in shorter, more focused sessions
Response: 5.

Scenario: I'm having trouble focusing on my studies because of distractions at home. What should I do?
Relevant Information: The user has tried studying in shorter, more focused sessions, but distractions persist.
Options:
    1: Suggest that the user studies with the TV on
    2: Suggest that the user gives up studying until they find a perfect environment
    3: Suggest that the user listens to music to drown out the distractions
    4: Suggest that the user moves to a different location to study
    5: Suggest that the user studies in shorter, more focused sessions
Response: 4.
""" 


# test 2 4 6 

def intention_prompt_second_fewshotlearning(scenario,  options, adapt_outcome , num_ex    ):
    """ Second adaptive prompt, same as first but also give the 'adapt_outcome' to the model in the system prompts
    """
    if num_ex == 2:
        examples = few_shotexamples
    elif num_ex == 4:
        examples = few_shotexamples + examples_second_2
        print('4 examples made')
    elif num_ex == 6:   
        examples = few_shotexamples + examples_second_2 + examples_third_2
        print('6 examples made')
    else:
        print('Error - wrong number of examples asked for')
    
  

    op1, op2, op3, op4 , op5= options
    system_prompt_content  = f"""You are a helpful and harmless assistant. You can only answer with numeric responses. Please respond with a number."""
    user_prompt_content = f"""Pay attention to the examples of your previous conversations: {examples}.
    Now answer the following scenario: 
    {scenario} 
    Relevant information: {adapt_outcome}.
    
    {op1}
    {op2}
    {op3}
    {op4}
    {op5}

    Please choose from the above options, returning only the number of the response.
    The correct answer: """ 
    
    prompt = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_prompt_content}
    ]
    
    return prompt 

# TODO steps of reasoning
chain_examples = """ 
Scenario: I have an upcoming exam and I am feeling very nervous about it, what should I do? 
Options:
    1: Suggest that the user dwells on their nerves
    2: Suggest that the user meditate or do a relaxing task
    3: Suggest that the user keeps to their normal schedule
    4: Suggest that the user learns to skateboard
    5: Suggest that the user takes a deep breath and reviews their notes

Answer: To address the feeling of nervousness about an upcoming exam, it's important to consider strategies that directly impact both emotional well-being and exam preparedness. 
Dwelling on nerves could increase anxiety and negatively affect exam performance, making it an unproductive choice. Meditating or doing a relaxing task can help reduce stress, improve focus, and enhance mental clarity.
Keeping to a normal schedule maintains routine but might not address the acute stress and anxiety directly associated with the upcoming exam.
Learning to skateboard introduces a new activity that, while potentially fun and distracting, does not directly contribute to reducing exam-related nervousness or improving exam readiness.
Taking a deep breath and reviewing notes is a good strategy for last-minute review and stress reduction. However, it might not be as effective in managing deeper feelings of nervousness as a dedicated period of meditation or relaxation.
Given these considerations, Option 2: Suggesting that the user meditate or do a relaxing task stands out as the most effective approach. It provides a direct method to alleviate nervousness by engaging in activities known to reduce stress and improve mental well-being. Meditation, specifically, is backed by research as a way to lower stress levels and enhance focus, making it an ideal strategy in the context of exam preparation where both emotional and cognitive performance are crucial.
Response:2.

Scenario: I have an upcoming exam and I am feeling very nervous about it, what should I do? Relevant Information: The user has already meditated or done a relaxing task. 
Options: 
    1: Suggest that the user dwells on their nerves
    2: Suggest that the user meditate or do a relaxing task
    3: Suggest that the user keeps to their normal schedule
    4: Suggest that the user learns to skateboard
    5: Suggest that the user takes a deep breath and reviews their notes
Answer: 
To address the feeling of nervousness about an upcoming exam, it's important to consider strategies that directly impact both emotional well-being and exam preparedness. 
Dwelling on nerves could increase anxiety and negatively affect exam performance, making it an unproductive choice. Meditating or doing a relaxing task can help reduce stress, improve focus, and enhance mental clarity.
Keeping to a normal schedule maintains routine but might not address the acute stress and anxiety directly associated with the upcoming exam.
Learning to skateboard introduces a new activity that, while potentially fun and distracting, does not directly contribute to reducing exam-related nervousness or improving exam readiness.
Taking a deep breath and reviewing notes is a good strategy for last-minute review and stress reduction. However, it might not be as effective in managing deeper feelings of nervousness as a dedicated period of meditation or relaxation.
Considering that the user has already attempted meditation and relaxation without fully alleviating their nervousness, the most effective next step would be to combine a calming technique with a practical study strategy.
Option 3, keeping to a normal schedule, offers structure and could help mitigate nerves by maintaining a sense of routine. However, Option 5, taking a deep breath and reviewing notes, directly addresses the exam preparation and provides an immediate, actionable step that combines calming techniques with effective study practices.
Response:5.


""" 

def intention_prompt_second_chainofthought(scenario, options, adapt_outcome ):
    print('Starting C-o-T') 

    op1, op2, op3, op4 , op5= options
    system_prompt_content  = f"""You are a helpful and harmless assistant. End your response with Response:X"""
    user_prompt_content = f"""
    Answer the following scenario, reason using chain of thought reasoning and then return Response: X  where X is the number of the chosen option. Pay attention to examples of previous conversations: {chain_examples}. Now answer the following scenario: 
    {scenario} 
    Relevant information: {adapt_outcome}.
    
    {op1}
    {op2}
    {op3}
    {op4}
    {op5}

    Please choose from the above options, returning your reasoning (less than 100 words) of the response and the number. In the format:
    Reasoning with Chain of Thought (less than 100 words): 
    Response: """ 
    
    prompt = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_prompt_content}
    ]
    
    return prompt 

