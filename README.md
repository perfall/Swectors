'''html
<!doctype html>
<html>
	<head>
	<title>Swectors</title>
	<meta charset="utf-8">
	<style>
	@import "http://fonts.googleapis.com/css?family=Open Sans:regular,semibold,italic,italicsemibold|Inconsolata&amp;v2";
	body {
  		font-family: "Open Sans";
  		margin: 2em 2em 2em 2em;
	}
	pre {
  		background: #eee;
  		margin-left: 2em;
  		padding: 5px;
  		border-left: 3px solid #ccc;
	}
</style>
</head>

<body>
	<h1>Swectors</h1>
	<p>This is the code used for the paper "Towards a Standard Dataset of Swedish Word Vectors", pdf can be found <a href="https://link.to.article">here</a>.</p>
	<h2>Creating Swedish word vectors</h2>
	<p>The script takes four parameters, method (cbow or sgns), dimensionality, window size and iterations.</p>
	<p>Such as:</p>
	<pre><code>python3 create_vectors.py cbow 300 10 40</code></pre>
	<pre><code>python3 create_vectors.py sgns 50 10 5</code></pre>
	<p>A textfile is created where each the first index of each row is a unique word and the rest of the row is each element of the vector, separated by spaces.</p>
	<h2>Data</h2>
	<p>The training set is located in 'sentences', each row corresponds to one sentence. Included is a sample of Göteborgsposten-2013 (100k rows).</p> 
	<h2>Evaluation</h2>
	Tools and instructions to how the use QVEC-CCA can be found <a href="https://github.com/ytsvetko/qvec">here</a>. For a quick start, simply download the file and add 'suc.saldo', then use the following line to evalute a set a vectors.
	<pre><code>./qvec_cca.py --in_vectors /path/to/vecs --in_oracle suc.saldo</code></pre>
	<h2>Contact</h2>
	<p>
		<a href="mailto:perfa292@student.liu.se?Subject=Swectors">Per Fallgren</a>
	</p>
	<p>
		<a href="mailto:jesse317@student.liu.se?Subject=Swectors">Jesper Segeblad</a>
	</p>
	<p>
		<a href="mailto:marco.kuhlmann@liu.se?Subject=Swectors">Marco Kuhlmann</a>
	</p>
</body>
</html>
'''