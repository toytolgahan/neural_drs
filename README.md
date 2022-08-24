<h1>text2drs NEURAL SEMANTIC PARSER</h1>
<h2>The project</h2>
The general aim is to convert natural language sentences into formal language expressions. There are two philosophical motivations behind this project. 
<ol>
    <li>to show that formal meaning representations can be generated by a purely connectionist model. So, they are not what drives the human language understanding system, but only a product of that system.</li>
    <li>to represent the <i>approximately</i> systematic and rigid aspects of natural language semantics, in addition to representing the flexible side of natural language semantics that a connectionist system provides.</li>
</ol>
<h3>Note</h3><p>Why do I defend purely connectionist model? <a href="https://open.metu.edu.tr/handle/11511/96733"> A DEFENSE OF MEANING ELIMINATIVISM: A CONNECTIONIST APPROACH [Ph.D. - Doctoral Program]. Middle East Technical University.</a></p>

<h2>DRT</h2>
DRT is a meaning representation formalism which is found more useful than first-order logic representations. In representing two semantic phenomena, it is advantageous over FOL.
<ul>
    <li>quantifier-variable bindings </li>
    <li>presuppositions</li>
</ul>

<h2>Experiments</h2>
You may simply run <code> bash ./pipelines/main.sh </code>
Here is the pipeline:
<ul>
    <li><code> bash ./pipelines/download_data.sh</code> &rarr; The data is generated from the Gröningen Meaning Bank</li>
    <li><code>python ./src/raw2data.py</code> &rarr; It is converted to input-output expressions through raw2text.py</li>
    <li><code>python ./src/pickles_data.py</code> &rarr; Converted to input sentence encodings and a set of contextual vectors for target DRS expressions through pickles-data.py</li>
    <li><code>python3 ./src/text2drs.py</code> &rarr; Trained to maximize the cosine similarity between output vectors and target vectors </li>
</ul>
    
    
<h2> The encoder </h2>
<div style="display:flex; ">
    <div style="padding:2px; margin: 2% 2% 2% 0%;"> word sequence </div>
    <div style="margin: 2% 0% 2% 0%;"> &rarr;</div>
    <div style="border:1px solid #000000; padding:2px; margin: 2% 2% 2% 0%;">Embeddings</div>
    <div style="padding:2px; margin: 2% 0% 2% 0%;"> &rarr;</div>
    <div style="border:1px solid #000000; padding:2px; margin: 2% 2% 2% 0%;">Dropout</div>
    <div style="padding:2px; margin: 2% 0% 2% 0%;"> &rarr;</div> 
    <div style="border:1px solid #000000; padding:2px; margin: 2% 2% 2% 0%;">LSTM</div>
    <div style="padding:2px; margin: 2% 0% 2% 0%;"> &rarr;</div> 
    <div style="padding:2px; margin: 2% 2% 2% 0%;"> context </div>
</div>

    
    
<h2>The decoder </h2>
<div style="display:flex; padding:2%; ">
    <div>
        <div style="border:1px solid; border-radius: 45%; padding:2px;  margin: 2% 2% 2% 0%;"> context </div>
        <div style="border:1px solid; border-radius: 45%; padding:2px;  margin: 2% 2% 2% 0%;"> start token </div> 
    </div>
    <div style="margin: 2% 0% 2% 0%;"> &rarr;</div>
    <div style="border:1px solid #000000; padding:2px; margin: 2% 2% 2% 0%;">Dropout</div>
    <div style="padding:2px; margin: 2% 0% 2% 0%;"> &rarr;</div>
    <div style="border:1px solid #000000; padding:2px; margin: 2% 2% 2% 0%;">LSTM</div>
    <div style="padding:2px; margin: 2% 0% 2% 0%;"> &rarr;</div> 
    <div style="padding:2px; margin: 2% 2% 2% 0%;"> next token </div>
</div>
    
    
<h2> The Dataset </h2>
<div><h3>Input:</h3> A text from Gröningen Meaning Bank
    <div style="border: 1px solid black; padding:2%;">
        The American Diabetes Association reports the disease is the leading cause of
new cases of blindness among adults. It is the leading cause of kidney failure.
The rate of amputation is 10 times higher among those who suffer from the
disease.

Experts say those who learn how to manage the disease early, can live healthier
and more normal lives. VOA's June Soh found camps that provide children with
this chronic disease a positive approach to living with diabetes while letting
them just be kids. Amy Katz Narrates. 
    </div>
</div>
<div>
    <h3>Target:</h3> Corresponding Discourse Representation Structure from the Gröningen Meaning Bank
    <div style="border: 1px solid black; padding:2%; overflow: scroll;height:30rem;">
        <ul>
            <li>discourse 1 t5 now</li>
            <li>discourse 1 x3 chrysler</li>
            <li>discourse 1 x1 neuter</li>
            <li>discourse 1 x4 mexico</li>
            <li>discourse 1 x1 u.s.</li>
            <li>discourse 1 x1 x3 nn</li>
            <li>discourse 1 x2 automaker</li>
            <li>discourse 1 x2 x3 nn</li>
            <li>discourse 1 x4 northern</li>
            <li>discourse 1 x6 mexican_president_felipe_calderon</li>
            <li>discourse 1 x7 plant</li>
            <li>discourse 1 x7 x3 of</li>
            <li>discourse 1 x7 sixth</li>
            <li>discourse 1 x8 startup</li>
            <li>discourse 1 x8 ceremonial</li>
            <li>discourse 1 x9 saltillo</li>
            <li>discourse 1 x9 x10 nn</li>
            <li>discourse 1 x10 plant</li>
            <li>discourse 1 x11 mister</li>
            <li>discourse 1 x11 calderon</li>
            <li>discourse 1 x12 auto</li>
            <li>discourse 1 x12 x13 nn</li>
            <li>discourse 1 x13 industry</li>
            <li>discourse 1 x14 pentastar_v-6</li>
            <li>discourse 1 x14 x15 nn</li>
            <li>discourse 1 x15 engine</li>
            <li>discourse 1 x15 x1 of</li>
            <li>discourse 1 x15 new</li>
            <li>discourse 1 x15 fuel-efficient</li>
            <li>discourse 1 x16 dodge</li>
            <li>discourse 1 x17 jeep</li>
            <li>discourse 1 e21 build</li>
            <li>discourse 1 x19 engine</li>
            <li>discourse 1 x20 year</li>
            <li>discourse 1 x18 capacity</li>
            <li>discourse 1 x19</li>
            <li>discourse 1 x19 x20 per</li>
            <li>discourse 1 e21 x18 Agent</li>
            <li>discourse 1 e21 x19 Product</li>
            <li>discourse 1 x25 new</li>
            <li>discourse 1 x25 plant</li>
            <li>discourse 1 x26 dollar</li>
            <li>discourse 1 x24 engine</li>
            <li>discourse 1 x24 x25 nn</li>
            <li>discourse 1 x26</li>
            <li>discourse 1 x25 x26 rel</li>
            <li>discourse 1 e27 open</li>
            <li>discourse 1 x25 x4 in</li>
            <li>discourse 1 e27 x3 Agent</li>
            <li>discourse 1 e27 x25 Theme</li>
            <li>discourse 1 x28 t5</li>
            <li>discourse 1 x28 t29 temp_included</li>
            <li>discourse 1 e27 t29 temp_abut</li>
            <li>discourse 1 e31 say</li>
            <li>discourse 1 e31 x6 Agent</li>
            <li>discourse 1 e31 p32 Topic</li>
            <li>discourse 1 p32</li>
            <li>discourse 1 e31 t33 temp_included</li>
            <li>discourse 1 t33 t5 temp_before</li>
            <li>discourse 1 x10 friday</li>
            <li>discourse 1 x8 x10 of</li>
            <li>discourse 1 e31 x8 during</li>
            <li>discourse 1</li>
            <li>discourse 1 e35 create</li>
            <li>discourse 1 x34 job</li>
            <li>discourse 1 x7 x4 in</li>
            <li>discourse 1 x34</li>
            <li>discourse 1 e35 x7 Agent</li>
            <li>discourse 1 e35 x34 Product</li>
            <li>discourse 1 e35 t36 temp_included</li>
            <li>discourse 1 t5 t36 temp_before</li>
            <li>discourse 1 e38 say</li>
            <li>discourse 1 e38 x11 Agent</li>
            <li>discourse 1 e38 p39 Topic</li>
            <li>discourse 1 p39</li>
            <li>discourse 1 e38 t40 temp_included</li>
            <li>discourse 1 t40 t5 temp_before</li>
            <li>discourse 1 x41 leader</li>
            <li>discourse 1 x41 worldwide</li>
            <li>discourse 1 e42 become</li>
            <li>discourse 1 x41 x13 in</li>
            <li>discourse 1 e42 x4 agent</li>
            <li>discourse 1 e42 x41 patient</li>
            <li>discourse 1 x43 t5</li>
            <li>discourse 1 x43 t44 temp_included</li>
            <li>discourse 1 e42 t44 temp_abut</li>
            <li>discourse 1 e46 plan</li>
            <li>discourse 1 e46 x3 Experiencer</li>
            <li>discourse 1 e46 p47 Theme</li>
            <li>discourse 1 p47</li>
            <li>discourse 1 e46 t48 temp_included</li>
            <li>discourse 1 t48 t5</li>
            <li>discourse 1 e51 build</li>
            <li>discourse 1 x49 ram</li>
            <li>discourse 1 x49 x50 nn</li>
            <li>discourse 1 x50 vehicle</li>
            <li>discourse 1 x16 x17 rel</li>
            <li>discourse 1 x16 x50 rel</li>
            <li>discourse 1 x3 x16 rel</li>
            <li>discourse 1 x15 x3 for</li>
            <li>discourse 1 e51 x3 Agent</li>
            <li>discourse 1 e51 x15 Product</li>
            <li>discourse 1 e54 say</li>
            <li>discourse 1 x53 official</li>
            <li>discourse 1 x53 mexican</li>
            <li>discourse 1 e54 x53 Agent</li>
            <li>discourse 1 e54 p55 Topic</li>
            <li>discourse 1 p55</li>
            <li>discourse 1 e54 t56 temp_included</li>
            <li>discourse 1 t56 t5</li>
            <li>discourse 1</li>
            <li>discourse 1 e57 have</li>
            <li>discourse 1 e57 x25 Agent</li>
            <li>discourse 1 e57 x18 Patient</li>
            <li>discourse 1 e57 t58 temp_included</li>
            <li>discourse 1 t5 t58 temp_before</li>
            <li>discourse 2 x25 new</li>
            <li>discourse 2 x25 plant</li>
            <li>discourse 2 x26 dollar</li>
            <li>discourse 2 x24 engine</li>
            <li>discourse 2 x24 x25 nn</li>
            <li>discourse 2 x26</li>
            <li>discourse 2 x25 x26 rel</li>
            <li>discourse 2 e27 open</li>
            <li>discourse 2 x25 x4 in</li>
            <li>discourse 2 e27 x3 Agent</li>
            <li>discourse 2 e27 x25 Theme</li>
            <li>discourse 2 x28 t5</li>
            <li>discourse 2 x28 t29 temp_included</li>
            <li>discourse 2 e27 t29 temp_abut</li>
            <li>discourse 2 e31 say</li>
            <li>discourse 2 e31 x6 Agent</li>
            <li>discourse 2 e31 p32 Topic</li>
            <li>discourse 2 p32</li>
            <li>discourse 2 e31 t33 temp_included</li>
            <li>discourse 2 t33 t5 temp_before</li>
            <li>discourse 2 x10 friday</li>
            <li>discourse 2 x8 x10 of</li>
            <li>discourse 2 e31 x8 during</li>
            <li>discourse 2</li>
            <li>discourse 2 e35 create</li>
            <li>discourse 2 x34 job</li>
            <li>discourse 2 x7 x4 in</li>
            <li>discourse 2 x34</li>
            <li>discourse 2 e35 x7 Agent</li>
            <li>discourse 2 e35 x34 Product</li>
            <li>discourse 2 e35 t36 temp_included</li>
            <li>discourse 2 t5 t36 temp_before</li>
            <li>discourse 2 e38 say</li>
            <li>discourse 2 e38 x11 Agent</li>
            <li>discourse 2 e38 p39 Topic</li>
            <li>discourse 2 p39</li>
            <li>discourse 2 e38 t40 temp_included</li>
            <li>discourse 2 t40 t5 temp_before</li>
            <li>discourse 2 x41 leader</li>
            <li>discourse 2 x41 worldwide</li>
            <li>discourse 2 e42 become</li>
            <li>discourse 2 x41 x13 in</li>
            <li>discourse 2 e42 x4 agent</li>
            <li>discourse 2 e42 x41 patient</li>
            <li>discourse 2 x43 t5</li>
            <li>discourse 2 x43 t44 temp_included</li>
            <li>discourse 2 e42 t44 temp_abut</li>
            <li>discourse 2 e46 plan</li>
            <li>discourse 2 e46 x3 Experiencer</li>
            <li>discourse 2 e46 p47 Theme</li>
            <li>discourse 2 p47</li>
            <li>discourse 2 e46 t48 temp_included</li>
            <li>discourse 2 t48 t5</li>
            <li>discourse 2 e51 build</li>
            <li>discourse 2 x49 ram</li>
            <li>discourse 2 x49 x50 nn</li>
            <li>discourse 2 x50 vehicle</li>
            <li>discourse 2 x16 x17 rel</li>
            <li>discourse 2 x16 x50 rel</li>
            <li>discourse 2 x3 x16 rel</li>
            <li>discourse 2 x15 x3 for</li>
            <li>discourse 2 e51 x3 Agent</li>
            <li>discourse 2 e51 x15 Product</li>
            <li>discourse 2 e54 say</li>
            <li>discourse 2 x53 official</li>
            <li>discourse 2 x53 mexican</li>
            <li>discourse 2 e54 x53 Agent</li>
            <li>discourse 2 e54 p55 Topic</li>
            <li>discourse 2 p55</li>
            <li>discourse 2 e54 t56 temp_included</li>
            <li>discourse 2 t56 t5</li>
            <li>discourse 2</li>
            <li>discourse 2 e57 have</li>
            <li>discourse 2 e57 x25 Agent</li>
            <li>discourse 2 e57 x18 Patient</li>
            <li>discourse 2 e57 t58 temp_included</li>
            <li>discourse 2 t5 t58 temp_before</li>
        </ul>
    </div>
    </div>
    
    
    
    
    <h2>Contextual target vectors</h2>
<div style="display:flex; padding:2%;">
    <div style="border: 1px dashed white; text-align:center; padding:2% 2% 0% 0%; color:gray;">Gröningen <br> Meaning <br> Bank </div>
    <div style="border: 1px dashed white; text-align:center; padding:4% 2% 0% 0%; color:gray;"> &rarr;</div>
    <div style="border: 1px solid gray; text-align:center; padding:4%; color:gray;">xml Parser </div>
    <div style="border: 1px dashed white; text-align:center; padding:4% 2% 0% 0%;"> &rarr;</div>
    <div style="border: 1px dashed white; text-align:center; padding:2% 2% 0% 0%;">DRS <br> + <br> Text </div>
    <div style="border: 1px dashed white; text-align:center; padding:4% 2% 0% 0%;"> &rarr;</div>
    <div style="border: 1px solid black; text-align:center; padding:2%;">Transformer <br/> nlp = spacy.load("en_core_web_trf") </div>
    <div style="border: 1px dashed white; text-align:center; padding:4% 0% 0% 2%;"> &rarr;</div>
    <div style="border: 1px solid white; text-align:center;padding:2% 0% 0% 0%; margin:2%;">Contextual vectors (Concepts)</div>
</div>
<div>
    <h3>The idea behind contextual target vectors </h3>
    <p> The reason I use contextual vectors instead of discrete symbols ("DISCOURSE", "NOW", "NEUTER", etc.) as targets is to capture the structural relations in the target space. </p>
    <p>
        In the softmax-based model, the size of the last vector is the vocabulary. 
    </div>

<h2>Output sequence</h2>
<p> yhat @ concepts.T results in the similarity matrix <br/> then get the index of maximum elements </p>
<div style="display:flex; padding:5px;">
    <div style="font-size:40px; padding:5px;"> ( </div>
    <div style="padding: 0.5% 0 0 0;">yhat</div>
    <div style="font-size:40px; padding:5px;"> ) </div>
    <div style="font-size:40px; padding:5px;"> ( </div>
    <div style="padding: 0.5% 0 0 0;">concepts</div>
    <div style="font-size:40px; padding:5px;"> ) </div>
    <div style="padding: 0.5% 0 0 0;"> = </div>
    <div style="font-size:40px; padding:5px;"> ( </div>
    <div style="padding: 0.5% 0 0 0;">similarity matrix</div>
    <div style="font-size:40px; padding:5px;"> ) </div>
    <div style="padding: 0.5% 0 0 0;"> &rarr; </div>
    <div style="font-size:20px; padding:5px;"> [ </div>
    <div style="padding: 0.5% 0 0 0;">max elements indexes</div>
    <div style="font-size:20px; padding:5px;"> ] </div>
</div>
<h2>Needs revision!</h2>
The oncepts matrix needs to be revised for <u>two reasons</u>! 
<ol>
    <li>It is too big even for such a small model. 
        <ul>
            <li>
                Dimensionality reduction techniques, Fourier transformations, and autoencoders may be used to extract the key information.
            </li>
        </ul>
    </li>
    <li>Contextual embeddings extracted from a set of documents may not be sufficient to represent human concepts. Even if it is, it may be difficult to identify the concepts represented.
        <ul>
            <li>
                Distributed version of a concept dictionary (e.g. WordNET) may be an alternative..
            </li>
        </ul>
    </li>
    
