#TO RENDER:
#  Low quality --> manim -pql visualize.py CreateCircle
#  High quality --> manim -pqh visualize.py CreateCircle
# 
#SOURCE 
#       --> https://docs.manim.community/en/stable/_modules/manim/mobject/text/tex_mobject.html?highlight=subscript
#       --> https://www.3blue1brown.com/extras
#       --> https://github.com/3b1b/manim

from manim import *


class CreateCircle(Scene):
    def construct(self):
        equation = Text("equation")

        node1 = Circle().move_to([-5,3,0])  # create a circle  # set the color and transparency
        weight1 = Line().put_start_and_end_on([-4,3,0], [1,0,0])
        weight_label1 = Text("w1").next_to(weight1, UP)

        node2 = Circle().move_to([-5,0,0])
        weight2 = Line().put_start_and_end_on([-4,0,0], [1,0,0])

        node3 = Circle().move_to([-5,-3,0])
        weight3 = Line().put_start_and_end_on([-4,-3,0], [1,0,0])

        node4 = Circle().move_to([2,0,0])

        self.play(DrawBorderThenFill(node1), DrawBorderThenFill(node2), DrawBorderThenFill(node3), DrawBorderThenFill(node4))  # show the circle on screen
        self.play(DrawBorderThenFill(weight1), DrawBorderThenFill(weight2), DrawBorderThenFill(weight3))
        self.play(Create(weight_label1))
        self.play(Create(equation))
