#TO RENDER:
#  Low quality --> manim -pql visualize.py CreateCircle
#  High quality --> manim -pqh visualize.py CreateCircle
# 
#SOURCE 
#       --> https://docs.manim.community/en/stable/_modules/manim/mobject/text/tex_mobject.html?highlight=subscript
#       --> https://www.3blue1brown.com/extras
#       --> https://github.com/3b1b/manim

from manim import *



class Estructura(Scene):
    def construct(self):
        RADIUS = 0.3

        CIRCLE_COLOR = WHITE

        LAYER_1_X = -4
        LAYER_2_X = 0
        LAYER_3_X = 4

        # Title
        title = Text("Estructura de una Red Neuronal")

        # Input nodes
        node1 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_1_X,3,0])  # create a circle  # set the color and transparency

        node2 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_1_X,2,0])

        node3 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_1_X,1,0])

        node4 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_1_X,0,0])

        node5 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_1_X,-1,0])

        node6 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_1_X,-2,0])

        node7 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_1_X,-3,0])
        

        # Hidden layer
        node8 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_2_X,2,0])
        node9 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_2_X,1,0])
        node10 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_2_X,0,0])
        node11 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_2_X,-1,0])
        node12 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_2_X,-2,0])

        # Output layer
        node13 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_3_X,1,0])
        node14 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_3_X,0,0])
        node15 = Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_3_X,-1,0])

        # Rectangles to show layers

        input_layer = Rectangle(width=1.0, height=7.0, color=YELLOW).move_to([LAYER_1_X,0,0])
        hidden_layer = Rectangle(width=1.0, height=5.0, color=YELLOW).move_to([LAYER_2_X,0,0])
        output_layer = Rectangle(width=1.0, height=3.0, color=YELLOW).move_to([LAYER_3_X,0,0])

        # Show title
        self.play(DrawBorderThenFill(title))
        self.wait(3)
        self.play(FadeOut(title))

        # Show nodes
        self.play(DrawBorderThenFill(node1), DrawBorderThenFill(node2), DrawBorderThenFill(node3), DrawBorderThenFill(node4), DrawBorderThenFill(node5), DrawBorderThenFill(node6), DrawBorderThenFill(node7), DrawBorderThenFill(node8), DrawBorderThenFill(node9), DrawBorderThenFill(node10), DrawBorderThenFill(node11), DrawBorderThenFill(node12), DrawBorderThenFill(node13), DrawBorderThenFill(node14), DrawBorderThenFill(node15))  # show the circle on screen
        
        self.wait(12)

        # Show rectangles
        self.play(DrawBorderThenFill(input_layer))
        self.wait(3)
        self.play(FadeOut(input_layer))

        self.wait(1)

        self.play(DrawBorderThenFill(hidden_layer))
        self.wait(3)
        self.play(FadeOut(hidden_layer))

        self.wait(2)

        self.play(DrawBorderThenFill(output_layer))
        self.wait(1)
        self.play(FadeOut(output_layer))

        self.wait(2)

        # Draw Lines
        fist_layer_positions = [3,2,1,0,-1,-2,-3] 
        second_layer_positions = [2,1,0,-1,-2]
        third_layer_positions = [1,0,-1]

        First_Lines = list()
        for i in fist_layer_positions:
            for e in second_layer_positions:
                First_Lines.append(DrawBorderThenFill(Line(start=[LAYER_1_X+(RADIUS+0.08),i,0], end=[LAYER_2_X-(RADIUS+0.08),e,0]), run_time=3))


        Second_Lines = list()
        for i in second_layer_positions:
            for e in third_layer_positions:
                Second_Lines.append(DrawBorderThenFill(Line(start=[LAYER_2_X+(RADIUS+0.08),i,0], end=[LAYER_3_X-(RADIUS+0.08),e,0]), run_time=3))

        # Show lines
        self.play(*First_Lines)
        self.wait(1)

        self.play(*Second_Lines)
        self.wait(1)


class Formula(Scene):
    def construct(self):
        RADIUS = 0.5

        CIRCLE_COLOR = WHITE

        LAYER_1_X = -6
        LAYER_2_X = -2

        # Input nodes
        node1 = DrawBorderThenFill(Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_1_X,2,0]))  # create a circle  # set the color and transparency
        text1 = Create(Text("a1", font_size=20, color=RED).move_to([LAYER_1_X,2,0]))

        node2 = DrawBorderThenFill(Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_1_X,0,0]))
        text2 = Create(Text("a2", font_size=20, color=RED).move_to([LAYER_1_X,0,0]))

        node3 = DrawBorderThenFill(Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_1_X,-2,0]))
        text3 = Create(Text("a3", font_size=20, color=RED).move_to([LAYER_1_X,-2,0]))

        # Output node
        node4 = DrawBorderThenFill(Circle(RADIUS, color=CIRCLE_COLOR).move_to([LAYER_2_X,-0,0]))
        text4 = Create(Text("n", font_size=20, color=GREEN).move_to([LAYER_2_X,0,0]))

        fist_layer_positions = [2, 0, -2]
        second_layer_positions = [0]

        Lines = list()
        for i in fist_layer_positions:
            for e in second_layer_positions:
                Lines.append(DrawBorderThenFill(Line(start=[LAYER_1_X+(RADIUS+0.08),i,0], end=[LAYER_2_X-(RADIUS+0.08),e,0]), run_time=3))

        self.play(*[node1,node2,node3, node4, text1, text2, text3, text4])
        self.play(*Lines)

        form = Create(Text("n = sigmoid(a1*w1 + a2*w2 + a3*w3+ +b)", t2c={'a':RED, 'w':BLUE, 'n':GREEN}, font_size=25).move_to([3,2,0]), run_time=3)

        self.play(form)
        self.wait(5)
