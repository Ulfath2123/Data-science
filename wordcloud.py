from wordcloud import WordCloud
import matplotlib.pyplot as plt
text=If you dont like a test prompt, you can get a different (random) prompt with the "change test"
 button - or select a specific paragraph to type from the l
 ist below. To find out how fast you type, just start typing
 in the blank textbox on the right of the test prompt.
 You will see your progress, including errors on the left
 side as you type. In order to complete the test and save
 your score, you need to get 100% accuracy. You can fix
 errors as you go, or correct them at the end with the
 help of the spell checker.
word=WordCloud(width=800,height=400,background_color='white').generate(text)
plt.figure(figsize=(10,5))
plt.inshow(word,interpolatiom='bilinear')
plt.axis('off')
plt.show() 