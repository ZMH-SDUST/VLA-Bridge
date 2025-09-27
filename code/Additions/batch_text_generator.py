# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/15 14:32
@Auther ： Zzou
@File ：batch_text_generator.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import time

client = OpenAI(
    base_url="https://api.gpts.vin/v1",
    api_key=""
)

# Chinese vision
background = """
- Role: 动作识别专家
- Background: 用户需要对一系列动宾结构的动作文本进行简要扩建，扩建内容包括动作的简要介绍和人与物体的交互过程，要求客观描述，不加入主观意向，且生成的文本长度不超过50个字符。
- Profile: 你是一位在动作识别领域有着深厚经验的专家，擅长分析和描述人体动作及其与物体的交互。
- Skills: 动作分析、文本编写、客观描述、字符数控制。
- Goals: 设计一个能够准确、简洁地描述动作和交互的文本扩建流程。
- Constrains: 文本扩建必须客观，不包含主观意向，且长度不超过50个字符。
- OutputFormat: 简洁的英文文本描述。
- Workflow:
  1. 分析动作文本，确定主要动作和涉及的物体。
  2. 描述动作的物理过程和人与物体之间的交互。
  3. 确保描述客观，不加入主观意向，并控制字符数。
- Examples:
Q：Ride a bicycle
A：Riding a bicycle involves balance, coordination, and physical exertion. The rider mounts the bike, propels forward by pushing pedals with their feet. Steering is achieved by turning handlebars. Brakes slow or stop the bike. Helmets are worn for safety.
Q：Throw a frisbee
A：Throwing a frisbee involves grasping the disc, typically with a forehand grip, then swinging the arm and releasing the frisbee at the right moment for it to glide through the air. Direction and distance depend on the angle and speed ofthe throw. It’s a common recreational activity.
Q：Juggle oranges
A：Juggling oranges involwes tossing and catching multiple oranges in a specific pattern. Typically, the juggler keeps more oranges in the air than they havehands, lt reguires good hand-eye coordination, rhythm, and timing. This activity can be entertaining and challenging.
Q：Kickboxing
A:Kickboxine is a hvbrid combat soort that combines elements of ounchine from boxing and kickine from karate or muav tha. humans particinating ikickboxing engage in rigorous physical training, improving their strength,fiexibiity, and coordination. it's both a competitive sport and fitness regimen
Q:Negotiate with a human 
A：Negotiating with a human involwes communication and compromise to reach a mutual aereement, it reauires skis ike active istening, articuationpatience, and persuasion. Negotiations can occur in various contexts,including business transactions,confict resolution, and interpersonal relationships
Q: Sleep on a boat
A：Sleeping on a boat can be a unique experiece, influenced by factors like the boat's size, stabiity, and location. The gente rocking can be soothing, buconditions can vary, t's important to ensure safety precautions, such as wearing lifejackets, are in place.
Q：Shaking hands with a human
A：Shaking hands with a human is a common form of greeting, areement, or parting. The typical handshake involves a firm grip and brief up and dowrmovement. lt's a universal sign of goodwil, respect, and mutual understanding between individuals.
"""


def getdata(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": background},
            {"role": "user", "content": 'Q:' + text}
        ])
    return completion.choices[0].message.content


if __name__ == '__main__':

    with open('HOI_list.txt') as f:
        data = f.readlines()
    raw_list = []
    for i in tqdm(data):
        raw = getdata(i)
        raw_list.append(raw)
        time.sleep(0.1)

    df = pd.DataFrame({'Q': data, 'A': raw_list}, columns=['Q', 'A'])
    df.to_csv('output.csv', index=False)