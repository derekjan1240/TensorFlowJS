`use static`
// Load the binding
const tf = require('@tensorflow/tfjs-node');

// Tensors And Operations 
MainFun = async function(){
	console.log(`----------------------------------------------------------------------------\n`);
	/* Tensor */ 
	// Create Tensor
	let next = await CreateTensors();

	// Changing the shape of a Tensor
	next = await ReshapeTensor();
	
	// Getting values from a Tensor
	next = await GettingValuesFromTensor();

	/* Operations */
	next = await ManipulateData();

	/* Memory */
	next = await disposeTensor();

}()


/* Tensor */ 
async function CreateTensors(){
	// Create a rank-2 tensor (matrix) matrix tensor from a multidimensional array.
	let option = {
		array: [[1, 2, 3], [4, 5, 6]],
		shape: null,
		type: null
	}
	const a = await CreateTensor(option, '> Create Tensor from a multidimensional Array', false);

	option = {
		array: [1, 2, 3, 4],
		shape: [2, 2],
		type: null
	}
	const b = await CreateTensor(option, '\n> Create Tensor from a flat Array and Shape', false);

	// Tensors can also be created with bool, int32, complex64, and string dtypes
	option = {
		array: [[1, 2, 3], [4, 5, 6]],
		shape: [2, 3],
		type: 'int32'
	}
	const c = await CreateTensor(option, '\n> Create Tensor with Type');

	a.dispose();
	b.dispose();
	c.dispose();
}

async function ReshapeTensor(){
	option = {
		array: [1, 2, 3, 4, 5, 6],
		shape: null,
		type: null
	}
	const a = await CreateTensor(option, '> Original Tensor', false);
	const b = a.reshape([3, 2]);
	console.log(`\n> Changing the shape of Tensor`);
	console.log('New shape:', await b.shape);
	b.print();
	a.dispose();
	b.dispose();
	console.log(`----------------------------------------------------------------------------\n`);
};

async function GettingValuesFromTensor(){
	option = {
		array: [[1, 2], [3, 4]],
		shape: null,
		type: null
	}
	const a = await CreateTensor(option, '> The Tensor', false);
	
	// Returns the multi dimensional array of values.
	let arr = a.array();
	// Returns the flattened data that backs the tensor.
	let data = a.data();
	a.dispose();

	console.log(`\n> Getting values from the Tensor`);
	console.log('Arrat:', await arr, '\nData:', await data);
	console.log(`----------------------------------------------------------------------------\n`);
}

/* Operations */
async function ManipulateData(){
	console.log('> Operations');
	option = {
		array: [1, 2, 3, 4],
		shape: null,
		type: null
	}
	// create tensor x
	const x = await CreateTensor(option, 'Create Tensor x', false);

	const y = x.square();  // equivalent to tf.square(x)
	console.log('\ny = x.square()');
	y.print();
	
	console.log('\nz = x add y');
	const z = x.add(y);  
	z.print();

	console.log('\nmaximum(y, z)');
	const g = tf.maximum(y, z);
	g.print();
	console.log(`----------------------------------------------------------------------------\n`);
	
	x.dispose();
	y.dispose();
	z.dispose();
	g.dispose();
	// list of the operations 
	// tf.addN (tensors), tf.maximum(a, b), tf.minimum(a, b), tf.mod (a, b), tf.pow (base, exp), tf.squaredDifference(a, b) ...
	// https://js.tensorflow.org/api/latest/#Operations
}

/* Memory */
async function disposeTensor(){
	console.log('> Memory');
	
	// Create Tensor x
	option = {
		array: [[1, 2], [3, 4]],
		shape: null,
		type: null
	}
	const x = await CreateTensor(option, '> Create Tensor x', false);

	// Tidy > automatically be disposed
	const y = await tf.tidy(() => {
		const result = x.square().log().neg();
		console.log('\nBefore tidy:', tf.memory());     // tensor: x, y, x.square(), x.square().log() => numTensors: 4
		return result;
	});

	console.log('\nAfter tidy:', tf.memory());			// tensor: x, y => numTensors: 2

	// Dispose tensor x, y
	x.dispose();  // Equivalant to tf.dispose(a)
	tf.dispose(y);
	console.log('\nDispose all:', tf.memory());			// all tensor dispose => numTensors: 0

	console.log(`----------------------------------------------------------------------------\n`);
}


async function CreateTensor(option, msg, wall=true) {
	const tensor = await tf.tensor(option.array, option.shape, option.type);
	console.log(msg);
	console.log('option:', option);
	console.log(`shape: ${tensor.shape}   dtype: ${tensor.dtype}   size: ${tensor.size}`);
	tensor.print();
	wall? console.log(`----------------------------------------------------------------------------\n`): null;
	return tensor;
}
