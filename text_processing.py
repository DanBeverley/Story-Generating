import os

def text_combining(prompt , story , datapath):
    fp = open(os.path.join(datapath , prompt),encoding="utf8")
    fs = open(os.path.join(datapath , story) ,encoding="utf8")
    prompts = fp.readlines()
    stories = fs.readlines()
    assert len(prompts) == len(stories)
    combine = []
    for i in range(len(prompts)):
        combine.append(prompts[i].rstrip() + '<sep>' + " ".join(stories[i].split()[:300]))
    return combine

def text_cleaning(story):
    for p in '!,.:;?':
        story = story.replace(' ' + p, p)
    story = story.replace(' ' + "n't", "n't")
    story = story.replace(' ' + "'s", "'s")
    story = story.replace(' ' + "'re", "'re")
    story = story.replace(' ' + "'ve", "'ve")
    story = story.replace(' ' + "'ll", "'ll")
    story = story.replace(' ' + "'am", "'am")
    story = story.replace(' ' + "'m", "'m")
    return story