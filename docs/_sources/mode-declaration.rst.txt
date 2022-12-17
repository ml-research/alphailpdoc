Mode Declaration
================

alphaILP adopts *mode declaration* as a language bias.

Mode Declaration [Muggleton95, Ray07] is one of the common language
biases. We used mode declaration, which is defined as follows. A mode
declaration is either a head declaration
:math:`{\tt modeh(r, p(mdt_1, \ldots, mdt_n))}` or a body declaration
:math:`{\tt modeb(r, p(mdt_1, \ldots, mdt_n))}`, where
:math:`{\tt r}\in \mathbb{N}` is an integer, :math:`{\tt p}` is a
predicate, and :math:`{\tt mdt_i}` is a mode datatype. A mode datatype
is a tuple :math:`{\tt (pm, dt)}`, where :math:`{\tt pm}` is a
place-marker and :math:`{\tt dt}` is a datatype. A place-marker is
either :math:`\#`, which represents constants, or :math:`+` (resp.
:math:`−`), which represents input (resp. output) variables.
:math:`{\tt r}` represents the number of the usages of the predicate to
compose a solution.

-  Muggleton, S.: Inverse Entailment and Progol. New Generation
   Computing, Special issue on Inductive Logic Programming 13(3-4),
   245–286 (1995)
-  Ray, O., Inoue, K.: Mode-directed inverse entailment for full clausal
   theories. In: Proceedings of the 17th International Conference on
   Inductive Logic Programming (ILP). Lecture Notes in Computer Science,
   vol. 4894, pp. 225–238 (2007)

Mode declaration can be implemented using ``ModeDeclaration`` class in
``mode_declaration.py``.

For example, to solve Kandinsky patterns, the following mode
declarations are defined:

:math:`{\tt modeb(1, color(+object, \#color))}`

where :math:`{\tt modeb}` means that the declaration is for the body
atoms in clauses. This declaration can be read: *Atoms that have the
predicate of color can appear once in the body of clauses to be
generated. The first argument is a variable which has been already seen
on the clause (not to introduce a new variable), and the second is a
constant whose datatype is color.*

Using ``ModeTerm`` and ``DataType`` class, :math:`{\tt +object}` and
:math:`{\tt \#color}` can be instantiated:

::

       p_object = ModeTerm('+', DataType('object'))
       s_color = ModeTerm('#', DataType('color'))

Then mode declaration :math:`{\tt modeb(1, color(+object, \#color))}`
can be instantiated:
``ModeDeclaration('body', 1, lang.get_pred_by_name('color'), [p_object, s_color])``

alphaILP defines a serch space by having a list of mode declarations to
limit the clauses to be accepted as a candidate.
