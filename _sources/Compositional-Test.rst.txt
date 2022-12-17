Compositional Test by Differentiable Reasoning
==============================================

We briefly demostrate how we can achieve compositional test by using
differentiable reasoning. Suppose we have a buliding line of products in
an industrial company, and the company should check if all of the
necessary parts of the products are aligned in a correct manner before
sending them to customers.

We use a 3D visual environment
`CLEVR <https://cs.stanford.edu/people/jcjohns/clevr/>`__ to demonstrate
this task. Suppose we want to compose a product which should always
consist of large cube and large cylinder. Namely, the following iamges
show positive cases:

.. code:: ipython3

    from IPython.display import Image
    Image('imgs/clevr/clevrhans_positive.png')




.. image:: _static/output_compositional_0.png



On the contrary, the following examples should be detected as negative
cases, meaning that the product should be checked by humans because of
the error of its compositionality of necessary parts:

.. code:: ipython3

    Image('imgs/clevr/clevrhans_negative.png')




.. image:: _static/output_compositional_1.png



We realize an efficient compositionality checker from visual information
as a differentiable reasoner aided by expert knowledge.

Lanuage Definition
------------------

To start writing logic programs, we need to specify a set of symbols we
can use, which is called as **language**.

We define language in text files in
``data/lang/dataset-type/dataset-name/``. ### Predicates Predicates are
written in ``preds.txt`` file. The format is ``name:arity:data_types``.
Each predicate should be specified line by line. For example,

.. code:: prolog

   kp:1:image

Neural Predicates
~~~~~~~~~~~~~~~~~

Neural predicates are written in ``neural_preds.txt`` file. The format
is ``name:arity:data_types``. Each predicate should be specified line by
line. For example,

.. code:: prolog

   in:2:object,image
   color:2:object,color
   shape:2:object,shape
   size:2:object,size
   material:2:object,material

Valuation functions for each neural predicate should be defined in
``valuation_func.py`` and be registered in ``valuation.py``.

Constants
~~~~~~~~~

Constants are written in ``consts.txt``. The format is
``data_type:names``. Each constant should be specified line by line. For
example,

.. code:: prolog

   object:obj0,obj1,obj2,obj3,obj4,obj5,obj6,obj7,obj8,obj9
   color:cyan,blue,yellow,purple,red,green,gray,brown
   shape:sphere,cube,cylinder
   size:large,small
   material:rubber,metal
   image:img

The defined language can be loaded by ``logic_utils.get_lang``.

.. code:: ipython3

    # Load a defined language
    import sys
    sys.path.append('src/')
    from src.logic_utils import get_lang
    
    lark_path = 'src/lark/exp.lark'
    lang_base_path = 'data/lang/'
    lang, _clauses, bk_clauses, bk, atoms = get_lang(
            lark_path, lang_base_path, 'clevr', 'clevr-hans0')

Specify Hyperparameters
-----------------------

.. code:: ipython3

    import torch
    class Args:
        dataset_type = 'clevr'
        dataset = 'clevr-hans0'
        batch_size = 2
        num_objects = 6
        no_cuda = True
        num_workers = 4
        program_size = 1
        epochs = 20
        lr = 1e-2
        infer_step = 3
        term_depth = 2
        no_train = False
        plot = False
        small_data = False
    
    args = Args()
    device = torch.device('cpu')

Writing Logic Programs
----------------------

By using the defined symbols, you can write logic programs, for example,

.. code:: prolog

   kp(X):-in(O1,X),in(O2,X),size(O1,large),shape(O1,cube),size(O2,large),shape(O2,cylinder).

Clauses should be written in ``clauses.txt`` or ``bk_clauses.txt``.

.. code:: ipython3

    # Write a logic program as text
    clauses_str = """
    kp(X):-in(O1,X),in(O2,X),size(O1,large),shape(O1,cube),size(O2,large),shape(O2,cylinder).
    """
    # Parse the text to logic program
    from fol.data_utils import DataUtils
    du = DataUtils(lark_path, lang_base_path, args.dataset_type, args.dataset)
    clauses = []
    for line in clauses_str.split('\n')[1:-1]:
        clauses.append(du.parse_clause(line, lang))
        
    clauses = [clauses[0]]

Build a Reasoner
----------------

Import the neuro-symbolic forward reasoner.

.. code:: ipython3

    from percept import SlotAttentionPerceptionModule, YOLOPerceptionModule
    from valuation import SlotAttentionValuationModule, YOLOValuationModule
    from facts_converter import FactsConverter
    from nsfr import NSFReasoner
    from logic_utils import build_infer_module, build_clause_infer_module
    import torch
    
    PM = SlotAttentionPerceptionModule(e=args.num_objects, d=11, device=device)
    VM = SlotAttentionValuationModule(
                lang=lang, device=device)
    
    FC = FactsConverter(lang=lang, perception_module=PM,
                            valuation_module=VM, device=device)
    IM = build_infer_module(clauses, bk_clauses, atoms, lang,
                                m=1, infer_step=3, device=device, train=False)
    CIM = build_clause_infer_module(clauses, bk_clauses, atoms, lang,
                                m=len(clauses), infer_step=3, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(perception_module=PM, facts_converter=FC,
                           infer_module=IM, clause_infer_module=CIM, atoms=atoms, bk=bk, clauses=clauses)


.. parsed-literal::

    Pretrained  neural predicates have been loaded!


Load Data
---------

.. code:: ipython3

    from nsfr_utils import get_data_loader  # get torch data loader
    import matplotlib.pyplot as plt
    
    train_loader, val_loader,  test_loader = get_data_loader(args)

.. code:: ipython3

    from train import predict
    acc_th = predict(NSFR, train_loader, args, device, th=0.2)
    print('Accuracy: ', acc_th[0])


.. parsed-literal::

    27it [00:22,  1.17it/s]

.. parsed-literal::

    Accuracy:  0.9629629629629629


.. parsed-literal::

    


By performing differentiable reasoning on visual scenes, the task of
compositional test can be solved efficiently by alphaILP.
