======
README
======
----------------------------
Library for comparing sequence classifiers
----------------------------

Installation
============

Manual installation
-------------------

1. Download or clone this repository:

.. code-block:: bash

    git clone https://github.com/pmiara/Sequence-Classification.git

2. Go to downloaded directory:

.. code-block:: bash

    cd Sequence-Classification

3. Install dependencies:

.. code-block:: bash

    pip3 install -r requirements.txt

Docker
------
1. Install Docker and docker-compose on your machine.
2. Download or clone this repository:

.. code-block:: bash

    git clone https://github.com/pmiara/Sequence-Classification.git

3. Go to downloaded directory:

.. code-block:: bash

    cd Sequence-Classification

4. Run docker-compose:

.. code-block:: bash

    docker-compose up

5. If you add any new dependency you have to rebuild docker image. To do so type:

.. code-block:: bash

    docker image rm -f sequences
    docker-compose up

Google Colab
------------
1. Go to: https://colab.research.google.com/github/pmiara/Sequence-Classification/blob/master/usage_example.ipynb
2. Click 'COPY TO DRIVE'.
3. Click 'OPEN IN NEW TAB'.
4. Create new cell above and run this code in it:

.. code-block:: bash

    !git clone https://github.com/pmiara/Sequence-Classification
    !pip install -q mlxtend
    %cd Sequence-Classification

5. Run remaining cells. Don't run again the cell you've just added, because it would cause some errors.

Authors
=======
* Jakub Berezowski
* Magda Lipowska
* Piotr Miara
* Grzegorz Szczepaniak
* Maciej Piernik - supervisor

License
=======
This project is licensed under the MIT License - see the LICENSE file in the repository's main directory for details.
