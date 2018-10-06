
/**
 * INSTRUCTIONS.
 * 
 * Problem Set 3 will be extensive. You've been warned. Practice
 * makes perfect, and we'll get a lot of practice with conditional
 * and iterative control structures.
 * 
 * A few new concepts are being introduced with this assignment.
 * Last time, we saw methods (other than the main method) for the
 * first time. Now, we're seeing parameters, which are the variables
 * inside the parentheses in our methods. We'll cover these in more
 * detail later. For now, you just need to understand the basics.
 * 
 * A parameter is data that is provided to the method. It helps the
 * method do its job. It's passed in through the parentheses, and
 * you can reference these variables by name as if they're your own.
 * 
 * Each of the return statements included in this skeleton are
 * placeholders. You'll need to modify them as you write your code.
 */
public class ProblemSet3 {
	/**
	 * Make sure you're testing your code by calling your methods
	 * from the main method!
	 */

	

	public static void main(String[] args) {
			ProblemSet3 ps3 = new ProblemSet3();
			ps3.tests();
	}

	/*
	 * You and your date are trying to get a table at a restaurant. The
	 * parameter "you" is the stylishness of your clothes, in the range
	 * 0..10, and "date" is the stylishness of your date's clothes. The
	 * result getting the table is encoded as an int value with (0 = no,
	 * 1 = maybe, 2 = yes). If either of you is very stylish, 8 or more,
	 * then the result is 2 (yes). With the exception that if either of
	 * you has style of 2 or less, then the result is 0 (no). Otherwise
	 * the result is 1 (maybe). Print YES, NO, or MAYBE. Nothing more,
	 * nothing less.
	 * 
	 * dateFashion(5, 10) → YES
	 * dateFashion(5, 2) → NO
	 * dateFashion(5, 5) → MAYBE
	 */

	

	public void dateFashion(int you, int date) {
		if (you <= 2 || date <= 2) {
			System.out.println("NO.");
		}
		else if (you >= 8 || date >= 8) {
			System.out.println("YES.");
		}
		else {
			System.out.println("MAYBE.");
		}
	}

	/*
	 * Given a string str, if the string starts with "f" print FIZZ.
	 * If the string ends with "b" print BUZZ. If both the "f" and "b"
	 * conditions are true, print FIZZBUZZ. In all other cases, print
	 * the string unchanged.
	 * 
	 * fizzString("fig") → FIZZ
	 * fizzString("dib") → BUZZ
	 * fizzString("fib") → FIZZBUZZ
	 */

	

	public void fizzString(String str) {
		if (str.startsWith("f") && !str.endsWith("b")) {
			System.out.println("FIZZ.");
		} 
		else if (str.endsWith("b") && !str.startsWith("f")) {
			System.out.println("BUZZ.");
		}
		else if (str.startsWith("f") && str.endsWith("b")){
			System.out.println("BUZZ.");
		}
		else {
			System.out.println(str);
		}
	}

	

	/*
	 * The squirrels in Palo Alto spend most of the day playing. In particular,
	 * they play if the temperature is between 60 and 90 (inclusive). Unless it
	 * is summer, then the upper limit is 100 instead of 90. Given an int
	 * temperature and a boolean isSummer, print YES if the squirrels play and
	 * NO otherwise.
	 * 
	 * squirrelPlay(70, false) → YES
	 * squirrelPlay(95, false) → NO
	 * squirrelPlay(95, true) → YES
	 */

	

	public void squirrelPlay(int temp, boolean isSummer) {
		if (isSummer == true) {
			if (temp >= 60 && temp <= 100) {
				System.out.println("YES.");
			}
			else {
				System.out.println("NO.");
			}
		}

		else {
			if (temp >= 60 && temp <= 90) {
				System.out.println("YES.");
			}
			else {
				System.out.println("NO.");
			}
		}
	}

	

	/*
	 * Given an int n, print the string form of the number followed by "!". So the
	 * int 6 yields "6!". Except if the number is divisible by 3 print FIZZ instead of
	 * the number; if the number is divisible by 5 print BUZZ; if divisible by
	 * both 3 and 5, print FIZZBUZZ.
	 * 
	 * fizzStringAgain(1) → 1!
	 * fizzStringAgain(2) → 2!
	 * fizzStringAgain(3) → FIZZ!
	 */

	

	public void fizzStringAgain(int n) {
		if (n%3 == 0 && n%5 ==0) {
			System.out.println("FIZZBUZZ!");
		}
		else if (n%3 == 0) {
			System.out.println("FIZZ!");
		}
		else if (n%5 ==0) {
			System.out.println("BUZZ!");
		}
		else {
			System.out.println(n + "!");
		}
	}
	

	/*
	 * We want to make a row of bricks that is goal inches long. We have a number of small
	 * bricks (1 inch each) and big bricks (5 inches each). Print YES if it is possible
	 * to make the goal by choosing from the given bricks, otherwise print NO. This is a
	 * little harder than it looks and can be done without any loops.
	 * 
	 * makeBricks(3, 1, 8) → YES
	 * makeBricks(3, 1, 9) → NO
	 * akeBricks(3, 2, 10) → YES
	 */

	public void makeBricks(int small, int big, int goal) {
		if (big >= goal / 5) {
			big -= goal / 5;
			goal %= 5;
		}
		else {
			goal -= big * 5;
		}
		if (small >= goal) {
			System.out.println("YES.");
		}
		else {
			System.out.println("NO.");
		}
	}

	/*
	 * Given 3 int values, a b c, print their sum. However, if one of the values is the
	 * same as another of the values, it does not count towards the sum.
	 * 
	 * loneSum(1, 2, 3) → 6
	 * loneSum(3, 2, 3) → 2
	 * loneSum(3, 3, 3) → 0
	 */

	

	public void loneSum(int a, int b, int c) {
		if (a == b && b == c && a ==c) {
			System.out.println(0 + ".");
		}
		else if (b == c) {
			System.out.println(a + ".");
		}
		else if (a == c) {
			System.out.println(b + ".");
		}
		else if (a == b) {
			System.out.println(c + ".");
		}
		else {
			System.out.println(a + b + c + ".");
		}
	}
	/*
	 * Given 3 int values, a b c, return their sum. However, if one of the values is 13
	 * then it does not count towards the sum and values to its right do not count. So for
	 * example, if b is 13, then both b and c do not count.
	 * 
	 * luckySum(1, 2, 3) → 6
	 * luckySum(1, 2, 13) → 3
	 * luckySum(1, 13, 3) → 1
	 */

	

	public void luckySum(int a, int b, int c) {
		if (a == 13) {
			System.out.println(0 + ".");
		}
		else if (b == 13) {
			System.out.println(a + ".");
		}
		else if (c == 13) {
			System.out.println(a + b + ".");
		}
		else {
			System.out.println(a + b + c + ".");
		}
	}

	/*
	 * Given an integer value, n, compute the factorial of n. You are required to use a
	 * for loop to solve this exercise.
	 * 
	 * factorialFor(3) → 6
	 * factorialFor(4) → 24
	 * factorialFor(5) → 120
	 */

	

	public void factorialFor(int n) {
		int factorial = 1;
		for (int i = 1; i <= n; i++) {
			factorial *= i;
		}
		System.out.println(n + "! = " + factorial + ".");
	}

	

	/*
	 * Given an integer value, n, compute the factorial of n. You are required to use a
	 * while loop to solve this exercise. Your method should print n! = y (where n is the
	 * input and y is the output). Nothing more, nothing less.
	 *
	 * factorialWhile(3) → 6
	 * factorialWhile(4) → 24
	 * factorialWhile(5) → 120
	 */

	

	public void factorialWhile(int n) {
		int i = 1;
		int factorial = 1;
		while (i <= n) {
			factorial *= i;
			i++;
		}

		System.out.println(n + "! = " + factorial + ".");
	}

	/*
	 * Given an integer value, n, determine whether or not n is a prime number. Your method
	 * should either print PRIME or NOT PRIME. Nothing more, nothing less.
	 * 
	 * isPrime(11) → YES
	 * isPrime(17) → YES
	 * isPrime(112) → NO

	 */

	

	public void isPrime(int n) {

		boolean check = false;
		for(int i = 2; i <= n/2; i++) {

			if (n % i == 0) {
				check = true;

			} 

		}

		

		if(!check && n >= 2) {

			System.out.println("PRIME.");

		}

		else {

			System.out.println("NOT PRIME.");

		}

	}

	

	public void tests () {
		// dateFashion(you, date) Test Cases

		this.dateFashion(0, 0); // NO.

		this.dateFashion(0, 1); // NO.

		this.dateFashion(0, 2); // NO.

		this.dateFashion(0, 3); // NO.

		this.dateFashion(0, 4); // NO.

		this.dateFashion(0, 5); // NO.

		this.dateFashion(0, 6); // NO.

		this.dateFashion(0, 7); // NO.

		this.dateFashion(0, 8); // NO.

		this.dateFashion(0, 9); // NO.

		this.dateFashion(0, 10); // NO.

		this.dateFashion(1, 0); // NO.

		this.dateFashion(1, 1); // NO.

		this.dateFashion(1, 2); // NO.

		this.dateFashion(1, 3); // NO.

		this.dateFashion(1, 4); // NO.

		this.dateFashion(1, 5); // NO.

		this.dateFashion(1, 6); // NO.

		this.dateFashion(1, 7); // NO.

		this.dateFashion(1, 8); // NO.

		this.dateFashion(1, 9); // NO.

		this.dateFashion(1, 10); // NO.

		this.dateFashion(2, 0); // NO.

		this.dateFashion(2, 1); // NO.

		this.dateFashion(2, 2); // NO.

		this.dateFashion(2, 3); // NO.

		this.dateFashion(2, 4); // NO.

		this.dateFashion(2, 5); // NO.

		this.dateFashion(2, 6); // NO.

		this.dateFashion(2, 7); // NO.

		this.dateFashion(2, 8); // NO.

		this.dateFashion(2, 9); // NO.

		this.dateFashion(2, 10); // NO.

		this.dateFashion(3, 0); // NO.

		this.dateFashion(3, 1); // NO.

		this.dateFashion(3, 2); // NO.

		this.dateFashion(3, 3); // MAYBE.

		this.dateFashion(3, 4); // MAYBE.

		this.dateFashion(3, 5); // MAYBE.

		this.dateFashion(3, 6); // MAYBE.

		this.dateFashion(3, 7); // MAYBE.

		this.dateFashion(3, 8); // YES.

		this.dateFashion(3, 9); // YES.

		this.dateFashion(3, 10); // YES.

		this.dateFashion(4, 0); // NO.

		this.dateFashion(4, 1); // NO.

		this.dateFashion(4, 2); // NO.

		this.dateFashion(4, 3); // MAYBE.

		this.dateFashion(4, 4); // MAYBE.

		this.dateFashion(4, 5); // MAYBE.

		this.dateFashion(4, 6); // MAYBE.

		this.dateFashion(4, 7); // MAYBE.

		this.dateFashion(4, 8); // YES.

		this.dateFashion(4, 9); // YES.

		this.dateFashion(4, 10); // YES.

		this.dateFashion(5, 0); // NO.

		this.dateFashion(5, 1); // NO.

		this.dateFashion(5, 2); // NO.

		this.dateFashion(5, 3); // MAYBE.

		this.dateFashion(5, 4); // MAYBE.

		this.dateFashion(5, 5); // MAYBE.

		this.dateFashion(5, 6); // MAYBE.

		this.dateFashion(5, 7); // MAYBE.

		this.dateFashion(5, 8); // YES.

		this.dateFashion(5, 9); // YES.

		this.dateFashion(5, 10); // YES.

		this.dateFashion(6, 0); // NO.

		this.dateFashion(6, 1); // NO.

		this.dateFashion(6, 2); // NO.

		this.dateFashion(6, 3); // MAYBE.

		this.dateFashion(6, 4); // MAYBE.

		this.dateFashion(6, 5); // MAYBE.

		this.dateFashion(6, 6); // MAYBE.

		this.dateFashion(6, 7); // MAYBE.

		this.dateFashion(6, 8); // YES.

		this.dateFashion(6, 9); // YES.

		this.dateFashion(6, 10); // YES.

		this.dateFashion(7, 0); // NO.

		this.dateFashion(7, 1); // NO.

		this.dateFashion(7, 2); // NO.

		this.dateFashion(7, 3); // MAYBE.

		this.dateFashion(7, 4); // MAYBE.

		this.dateFashion(7, 5); // MAYBE.

		this.dateFashion(7, 6); // MAYBE.

		this.dateFashion(7, 7); // MAYBE.

		this.dateFashion(7, 8); // YES.

		this.dateFashion(7, 9); // YES.

		this.dateFashion(7, 10); // YES.

		this.dateFashion(8, 0); // NO.

		this.dateFashion(8, 1); // NO.

		this.dateFashion(8, 2); // NO.

		this.dateFashion(8, 3); // YES.

		this.dateFashion(8, 4); // YES.

		this.dateFashion(8, 5); // YES.

		this.dateFashion(8, 6); // YES.

		this.dateFashion(8, 7); // YES.

		this.dateFashion(8, 8); // YES.

		this.dateFashion(8, 9); // YES.

		this.dateFashion(8, 10); // YES.

		this.dateFashion(9, 0); // NO.

		this.dateFashion(9, 1); // NO.

		this.dateFashion(9, 2); // NO.

		this.dateFashion(9, 3); // YES.

		this.dateFashion(9, 4); // YES.

		this.dateFashion(9, 5); // YES.

		this.dateFashion(9, 6); // YES.

		this.dateFashion(9, 7); // YES.

		this.dateFashion(9, 8); // YES.

		this.dateFashion(9, 9); // YES.

		this.dateFashion(9, 10); // YES.

		this.dateFashion(10, 0); // NO.

		this.dateFashion(10, 1); // NO.

		this.dateFashion(10, 2); // NO.

		this.dateFashion(10, 3); // YES.

		this.dateFashion(10, 4); // YES.

		this.dateFashion(10, 5); // YES.

		this.dateFashion(10, 6); // YES.

		this.dateFashion(10, 7); // YES.

		this.dateFashion(10, 8); // YES.

		this.dateFashion(10, 9); // YES.

		this.dateFashion(10, 10); // YES.



		// fizzString(str) Test Cases



		this.fizzString("for"); // FIZZ.

		this.fizzString("fit"); // FIZZ.

		this.fizzString("fax"); // FIZZ.

		this.fizzString("fat"); // FIZZ.

		this.fizzString("fix"); // FIZZ.

		this.fizzString("job"); // BUZZ.

		this.fizzString("rob"); // BUZZ.

		this.fizzString("pub"); // BUZZ.

		this.fizzString("tab"); // BUZZ.

		this.fizzString("rib"); // BUZZ.

		this.fizzString("fab"); // FIZZBUZZ.

		this.fizzString("fib"); // FIZZBUZZ.

		this.fizzString("flab"); // FIZZBUZZ.

		this.fizzString("feeb"); // FIZZBUZZ.

		this.fizzString("forb"); // FIZZBUZZ.

		this.fizzString("qwerty"); // qwerty

		this.fizzString("asdfg"); // asdfg

		this.fizzString("mnbvc"); // mnbvc

		this.fizzString("qazwsx"); // qazwsx

		this.fizzString("edcrfv"); // edcrfv



		// squirrelPlay(temp, isSummer) Test Cases



		this.squirrelPlay(50, false); // NO.

		this.squirrelPlay(50, true); // NO.

		this.squirrelPlay(51, false); // NO.

		this.squirrelPlay(51, true); // NO.

		this.squirrelPlay(52, false); // NO.

		this.squirrelPlay(52, true); // NO.

		this.squirrelPlay(53, false); // NO.

		this.squirrelPlay(53, true); // NO.

		this.squirrelPlay(54, false); // NO.

		this.squirrelPlay(54, true); // NO.

		this.squirrelPlay(55, false); // NO.

		this.squirrelPlay(55, true); // NO.

		this.squirrelPlay(56, false); // NO.

		this.squirrelPlay(56, true); // NO.

		this.squirrelPlay(57, false); // NO.

		this.squirrelPlay(57, true); // NO.

		this.squirrelPlay(58, false); // NO.

		this.squirrelPlay(58, true); // NO.

		this.squirrelPlay(59, false); // NO.

		this.squirrelPlay(59, true); // NO.

		this.squirrelPlay(60, false); // YES.

		this.squirrelPlay(60, true); // YES.

		this.squirrelPlay(61, false); // YES.

		this.squirrelPlay(61, true); // YES.

		this.squirrelPlay(62, false); // YES.

		this.squirrelPlay(62, true); // YES.

		this.squirrelPlay(63, false); // YES.

		this.squirrelPlay(63, true); // YES.

		this.squirrelPlay(64, false); // YES.

		this.squirrelPlay(64, true); // YES.

		this.squirrelPlay(65, false); // YES.

		this.squirrelPlay(65, true); // YES.

		this.squirrelPlay(66, false); // YES.

		this.squirrelPlay(66, true); // YES.

		this.squirrelPlay(67, false); // YES.

		this.squirrelPlay(67, true); // YES.

		this.squirrelPlay(68, false); // YES.

		this.squirrelPlay(68, true); // YES.

		this.squirrelPlay(69, false); // YES.

		this.squirrelPlay(69, true); // YES.

		this.squirrelPlay(70, false); // YES.

		this.squirrelPlay(70, true); // YES.

		this.squirrelPlay(71, false); // YES.

		this.squirrelPlay(71, true); // YES.

		this.squirrelPlay(72, false); // YES.

		this.squirrelPlay(72, true); // YES.

		this.squirrelPlay(73, false); // YES.

		this.squirrelPlay(73, true); // YES.

		this.squirrelPlay(74, false); // YES.

		this.squirrelPlay(74, true); // YES.

		this.squirrelPlay(75, false); // YES.

		this.squirrelPlay(75, true); // YES.

		this.squirrelPlay(76, false); // YES.

		this.squirrelPlay(76, true); // YES.

		this.squirrelPlay(77, false); // YES.

		this.squirrelPlay(77, true); // YES.

		this.squirrelPlay(78, false); // YES.

		this.squirrelPlay(78, true); // YES.

		this.squirrelPlay(79, false); // YES.

		this.squirrelPlay(79, true); // YES.

		this.squirrelPlay(80, false); // YES.

		this.squirrelPlay(80, true); // YES.

		this.squirrelPlay(81, false); // YES.

		this.squirrelPlay(81, true); // YES.

		this.squirrelPlay(82, false); // YES.

		this.squirrelPlay(82, true); // YES.

		this.squirrelPlay(83, false); // YES.

		this.squirrelPlay(83, true); // YES.

		this.squirrelPlay(84, false); // YES.

		this.squirrelPlay(84, true); // YES.

		this.squirrelPlay(85, false); // YES.

		this.squirrelPlay(85, true); // YES.

		this.squirrelPlay(86, false); // YES.

		this.squirrelPlay(86, true); // YES.

		this.squirrelPlay(87, false); // YES.

		this.squirrelPlay(87, true); // YES.

		this.squirrelPlay(88, false); // YES.

		this.squirrelPlay(88, true); // YES.

		this.squirrelPlay(89, false); // YES.

		this.squirrelPlay(89, true); // YES.

		this.squirrelPlay(90, false); // YES.

		this.squirrelPlay(90, true); // YES.

		this.squirrelPlay(91, false); // NO.

		this.squirrelPlay(91, true); // YES.

		this.squirrelPlay(92, false); // NO.

		this.squirrelPlay(92, true); // YES.

		this.squirrelPlay(93, false); // NO.

		this.squirrelPlay(93, true); // YES.

		this.squirrelPlay(94, false); // NO.

		this.squirrelPlay(94, true); // YES.

		this.squirrelPlay(95, false); // NO.

		this.squirrelPlay(95, true); // YES.

		this.squirrelPlay(96, false); // NO.

		this.squirrelPlay(96, true); // YES.

		this.squirrelPlay(97, false); // NO.

		this.squirrelPlay(97, true); // YES.

		this.squirrelPlay(98, false); // NO.

		this.squirrelPlay(98, true); // YES.

		this.squirrelPlay(99, false); // NO.

		this.squirrelPlay(99, true); // YES.

		this.squirrelPlay(100, false); // NO.

		this.squirrelPlay(100, true); // YES.

		this.squirrelPlay(101, false); // NO.

		this.squirrelPlay(101, true); // NO.

		this.squirrelPlay(102, false); // NO.

		this.squirrelPlay(102, true); // NO.

		this.squirrelPlay(103, false); // NO.

		this.squirrelPlay(103, true); // NO.

		this.squirrelPlay(104, false); // NO.

		this.squirrelPlay(104, true); // NO.

		this.squirrelPlay(105, false); // NO.

		this.squirrelPlay(105, true); // NO.

		this.squirrelPlay(106, false); // NO.

		this.squirrelPlay(106, true); // NO.

		this.squirrelPlay(107, false); // NO.

		this.squirrelPlay(107, true); // NO.

		this.squirrelPlay(108, false); // NO.

		this.squirrelPlay(108, true); // NO.

		this.squirrelPlay(109, false); // NO.

		this.squirrelPlay(109, true); // NO.



		// fizzStringAgain(n) Test Cases



		this.fizzStringAgain(0); // FIZZBUZZ!

		this.fizzStringAgain(1); // 1!

		this.fizzStringAgain(2); // 2!

		this.fizzStringAgain(3); // FIZZ!

		this.fizzStringAgain(4); // 4!

		this.fizzStringAgain(5); // BUZZ!

		this.fizzStringAgain(6); // FIZZ!

		this.fizzStringAgain(7); // 7!

		this.fizzStringAgain(8); // 8!

		this.fizzStringAgain(9); // FIZZ!

		this.fizzStringAgain(10); // BUZZ!

		this.fizzStringAgain(11); // 11!

		this.fizzStringAgain(12); // FIZZ!

		this.fizzStringAgain(13); // 13!

		this.fizzStringAgain(14); // 14!

		this.fizzStringAgain(15); // FIZZBUZZ!

		this.fizzStringAgain(16); // 16!

		this.fizzStringAgain(17); // 17!

		this.fizzStringAgain(18); // FIZZ!

		this.fizzStringAgain(19); // 19!

		this.fizzStringAgain(20); // BUZZ!

		this.fizzStringAgain(21); // FIZZ!

		this.fizzStringAgain(22); // 22!

		this.fizzStringAgain(23); // 23!

		this.fizzStringAgain(24); // FIZZ!

		this.fizzStringAgain(25); // BUZZ!

		this.fizzStringAgain(26); // 26!

		this.fizzStringAgain(27); // FIZZ!

		this.fizzStringAgain(28); // 28!

		this.fizzStringAgain(29); // 29!

		this.fizzStringAgain(30); // FIZZBUZZ!

		this.fizzStringAgain(31); // 31!

		this.fizzStringAgain(32); // 32!

		this.fizzStringAgain(33); // FIZZ!

		this.fizzStringAgain(34); // 34!

		this.fizzStringAgain(35); // BUZZ!

		this.fizzStringAgain(36); // FIZZ!

		this.fizzStringAgain(37); // 37!

		this.fizzStringAgain(38); // 38!

		this.fizzStringAgain(39); // FIZZ!

		this.fizzStringAgain(40); // BUZZ!

		this.fizzStringAgain(41); // 41!

		this.fizzStringAgain(42); // FIZZ!

		this.fizzStringAgain(43); // 43!

		this.fizzStringAgain(44); // 44!

		this.fizzStringAgain(45); // FIZZBUZZ!

		this.fizzStringAgain(46); // 46!

		this.fizzStringAgain(47); // 47!

		this.fizzStringAgain(48); // FIZZ!

		this.fizzStringAgain(49); // 49!

		this.fizzStringAgain(50); // BUZZ!

		this.fizzStringAgain(51); // FIZZ!

		this.fizzStringAgain(52); // 52!

		this.fizzStringAgain(53); // 53!

		this.fizzStringAgain(54); // FIZZ!

		this.fizzStringAgain(55); // BUZZ!

		this.fizzStringAgain(56); // 56!

		this.fizzStringAgain(57); // FIZZ!

		this.fizzStringAgain(58); // 58!

		this.fizzStringAgain(59); // 59!

		this.fizzStringAgain(60); // FIZZBUZZ!

		this.fizzStringAgain(61); // 61!

		this.fizzStringAgain(62); // 62!

		this.fizzStringAgain(63); // FIZZ!

		this.fizzStringAgain(64); // 64!

		this.fizzStringAgain(65); // BUZZ!

		this.fizzStringAgain(66); // FIZZ!

		this.fizzStringAgain(67); // 67!

		this.fizzStringAgain(68); // 68!

		this.fizzStringAgain(69); // FIZZ!

		this.fizzStringAgain(70); // BUZZ!

		this.fizzStringAgain(71); // 71!

		this.fizzStringAgain(72); // FIZZ!

		this.fizzStringAgain(73); // 73!

		this.fizzStringAgain(74); // 74!

		this.fizzStringAgain(75); // FIZZBUZZ!

		this.fizzStringAgain(76); // 76!

		this.fizzStringAgain(77); // 77!

		this.fizzStringAgain(78); // FIZZ!

		this.fizzStringAgain(79); // 79!

		this.fizzStringAgain(80); // BUZZ!

		this.fizzStringAgain(81); // FIZZ!

		this.fizzStringAgain(82); // 82!

		this.fizzStringAgain(83); // 83!

		this.fizzStringAgain(84); // FIZZ!

		this.fizzStringAgain(85); // BUZZ!

		this.fizzStringAgain(86); // 86!

		this.fizzStringAgain(87); // FIZZ!

		this.fizzStringAgain(88); // 88!

		this.fizzStringAgain(89); // 89!

		this.fizzStringAgain(90); // FIZZBUZZ!

		this.fizzStringAgain(91); // 91!

		this.fizzStringAgain(92); // 92!

		this.fizzStringAgain(93); // FIZZ!

		this.fizzStringAgain(94); // 94!

		this.fizzStringAgain(95); // BUZZ!

		this.fizzStringAgain(96); // FIZZ!

		this.fizzStringAgain(97); // 97!

		this.fizzStringAgain(98); // 98!

		this.fizzStringAgain(99); // FIZZ!



		// makeBricks(small, big, goal) Test Cases



		this.makeBricks(3, 1, 8); // YES.

		this.makeBricks(3, 1, 9); // NO.

		this.makeBricks(3, 2, 10); // YES.

		this.makeBricks(3, 2, 8); // YES.

		this.makeBricks(3, 2, 9); // NO.

		this.makeBricks(6, 1, 11); // YES.

		this.makeBricks(6, 0, 11); // NO.

		this.makeBricks(1, 4, 11); // YES.

		this.makeBricks(0, 3, 10); // YES.

		this.makeBricks(1, 4, 12); // NO.

		this.makeBricks(3, 1, 7); // YES.

		this.makeBricks(1, 1, 7); // NO.

		this.makeBricks(2, 1, 7); // YES.

		this.makeBricks(7, 1, 11); // YES.

		this.makeBricks(7, 1, 8); // YES.

		this.makeBricks(7, 1, 13); // NO.

		this.makeBricks(43, 1, 46); // YES.

		this.makeBricks(40, 1, 46); // NO.

		this.makeBricks(40, 2, 47); // YES.

		this.makeBricks(40, 2, 50); // YES.

		this.makeBricks(40, 2, 52); // NO.

		this.makeBricks(22, 2, 33); // NO.

		this.makeBricks(0, 2, 10); // YES.

		this.makeBricks(1000000, 1000, 1000100); // YES.

		this.makeBricks(2, 1000000, 100003); // NO.

		this.makeBricks(20, 0, 19); // YES.

		this.makeBricks(20, 0, 21); // NO.

		this.makeBricks(20, 4, 51); // NO.

		this.makeBricks(20, 4, 39); // YES.



		// loneSum(a, b, c) Test Cases



		this.loneSum(0, 0, 0); // 0.

		this.loneSum(0, 0, 1); // 1.

		this.loneSum(0, 0, 2); // 2.

		this.loneSum(0, 0, 3); // 3.

		this.loneSum(0, 0, 4); // 4.

		this.loneSum(0, 0, 5); // 5.

		this.loneSum(0, 0, 6); // 6.

		this.loneSum(0, 0, 7); // 7.

		this.loneSum(0, 0, 8); // 8.

		this.loneSum(0, 0, 9); // 9.

		this.loneSum(0, 1, 0); // 1.

		this.loneSum(0, 1, 1); // 0.

		this.loneSum(0, 1, 2); // 3.

		this.loneSum(0, 1, 3); // 4.

		this.loneSum(0, 1, 4); // 5.

		this.loneSum(0, 1, 5); // 6.

		this.loneSum(0, 1, 6); // 7.

		this.loneSum(0, 1, 7); // 8.

		this.loneSum(0, 1, 8); // 9.

		this.loneSum(0, 1, 9); // 10.

		this.loneSum(0, 2, 0); // 2.

		this.loneSum(0, 2, 1); // 3.

		this.loneSum(0, 2, 2); // 0.

		this.loneSum(0, 2, 3); // 5.

		this.loneSum(0, 2, 4); // 6.

		this.loneSum(0, 2, 5); // 7.

		this.loneSum(0, 2, 6); // 8.

		this.loneSum(0, 2, 7); // 9.

		this.loneSum(0, 2, 8); // 10.

		this.loneSum(0, 2, 9); // 11.

		this.loneSum(0, 3, 0); // 3.

		this.loneSum(0, 3, 1); // 4.

		this.loneSum(0, 3, 2); // 5.

		this.loneSum(0, 3, 3); // 0.

		this.loneSum(0, 3, 4); // 7.

		this.loneSum(0, 3, 5); // 8.

		this.loneSum(0, 3, 6); // 9.

		this.loneSum(0, 3, 7); // 10.

		this.loneSum(0, 3, 8); // 11.

		this.loneSum(0, 3, 9); // 12.

		this.loneSum(0, 4, 0); // 4.

		this.loneSum(0, 4, 1); // 5.

		this.loneSum(0, 4, 2); // 6.

		this.loneSum(0, 4, 3); // 7.

		this.loneSum(0, 4, 4); // 0.

		this.loneSum(0, 4, 5); // 9.

		this.loneSum(0, 4, 6); // 10.

		this.loneSum(0, 4, 7); // 11.

		this.loneSum(0, 4, 8); // 12.

		this.loneSum(0, 4, 9); // 13.

		this.loneSum(0, 5, 0); // 5.

		this.loneSum(0, 5, 1); // 6.

		this.loneSum(0, 5, 2); // 7.

		this.loneSum(0, 5, 3); // 8.

		this.loneSum(0, 5, 4); // 9.

		this.loneSum(0, 5, 5); // 0.

		this.loneSum(0, 5, 6); // 11.

		this.loneSum(0, 5, 7); // 12.

		this.loneSum(0, 5, 8); // 13.

		this.loneSum(0, 5, 9); // 14.

		this.loneSum(0, 6, 0); // 6.

		this.loneSum(0, 6, 1); // 7.

		this.loneSum(0, 6, 2); // 8.

		this.loneSum(0, 6, 3); // 9.

		this.loneSum(0, 6, 4); // 10.

		this.loneSum(0, 6, 5); // 11.

		this.loneSum(0, 6, 6); // 0.

		this.loneSum(0, 6, 7); // 13.

		this.loneSum(0, 6, 8); // 14.

		this.loneSum(0, 6, 9); // 15.

		this.loneSum(0, 7, 0); // 7.

		this.loneSum(0, 7, 1); // 8.

		this.loneSum(0, 7, 2); // 9.

		this.loneSum(0, 7, 3); // 10.

		this.loneSum(0, 7, 4); // 11.

		this.loneSum(0, 7, 5); // 12.

		this.loneSum(0, 7, 6); // 13.

		this.loneSum(0, 7, 7); // 0.

		this.loneSum(0, 7, 8); // 15.

		this.loneSum(0, 7, 9); // 16.

		this.loneSum(0, 8, 0); // 8.

		this.loneSum(0, 8, 1); // 9.

		this.loneSum(0, 8, 2); // 10.

		this.loneSum(0, 8, 3); // 11.

		this.loneSum(0, 8, 4); // 12.

		this.loneSum(0, 8, 5); // 13.

		this.loneSum(0, 8, 6); // 14.

		this.loneSum(0, 8, 7); // 15.

		this.loneSum(0, 8, 8); // 0.

		this.loneSum(0, 8, 9); // 17.

		this.loneSum(0, 9, 0); // 9.

		this.loneSum(0, 9, 1); // 10.

		this.loneSum(0, 9, 2); // 11.

		this.loneSum(0, 9, 3); // 12.

		this.loneSum(0, 9, 4); // 13.

		this.loneSum(0, 9, 5); // 14.

		this.loneSum(0, 9, 6); // 15.

		this.loneSum(0, 9, 7); // 16.

		this.loneSum(0, 9, 8); // 17.

		this.loneSum(0, 9, 9); // 0.

		this.loneSum(1, 0, 0); // 1.

		this.loneSum(1, 0, 1); // 0.

		this.loneSum(1, 0, 2); // 3.

		this.loneSum(1, 0, 3); // 4.

		this.loneSum(1, 0, 4); // 5.

		this.loneSum(1, 0, 5); // 6.

		this.loneSum(1, 0, 6); // 7.

		this.loneSum(1, 0, 7); // 8.

		this.loneSum(1, 0, 8); // 9.

		this.loneSum(1, 0, 9); // 10.

		this.loneSum(1, 1, 0); // 0.

		this.loneSum(1, 1, 1); // 0.

		this.loneSum(1, 1, 2); // 2.

		this.loneSum(1, 1, 3); // 3.

		this.loneSum(1, 1, 4); // 4.

		this.loneSum(1, 1, 5); // 5.

		this.loneSum(1, 1, 6); // 6.

		this.loneSum(1, 1, 7); // 7.

		this.loneSum(1, 1, 8); // 8.

		this.loneSum(1, 1, 9); // 9.

		this.loneSum(1, 2, 0); // 3.

		this.loneSum(1, 2, 1); // 2.

		this.loneSum(1, 2, 2); // 1.

		this.loneSum(1, 2, 3); // 6.

		this.loneSum(1, 2, 4); // 7.

		this.loneSum(1, 2, 5); // 8.

		this.loneSum(1, 2, 6); // 9.

		this.loneSum(1, 2, 7); // 10.

		this.loneSum(1, 2, 8); // 11.

		this.loneSum(1, 2, 9); // 12.

		this.loneSum(1, 3, 0); // 4.

		this.loneSum(1, 3, 1); // 3.

		this.loneSum(1, 3, 2); // 6.

		this.loneSum(1, 3, 3); // 1.

		this.loneSum(1, 3, 4); // 8.

		this.loneSum(1, 3, 5); // 9.

		this.loneSum(1, 3, 6); // 10.

		this.loneSum(1, 3, 7); // 11.

		this.loneSum(1, 3, 8); // 12.

		this.loneSum(1, 3, 9); // 13.

		this.loneSum(1, 4, 0); // 5.

		this.loneSum(1, 4, 1); // 4.

		this.loneSum(1, 4, 2); // 7.

		this.loneSum(1, 4, 3); // 8.

		this.loneSum(1, 4, 4); // 1.

		this.loneSum(1, 4, 5); // 10.

		this.loneSum(1, 4, 6); // 11.

		this.loneSum(1, 4, 7); // 12.

		this.loneSum(1, 4, 8); // 13.

		this.loneSum(1, 4, 9); // 14.

		this.loneSum(1, 5, 0); // 6.

		this.loneSum(1, 5, 1); // 5.

		this.loneSum(1, 5, 2); // 8.

		this.loneSum(1, 5, 3); // 9.

		this.loneSum(1, 5, 4); // 10.

		this.loneSum(1, 5, 5); // 1.

		this.loneSum(1, 5, 6); // 12.

		this.loneSum(1, 5, 7); // 13.

		this.loneSum(1, 5, 8); // 14.

		this.loneSum(1, 5, 9); // 15.

		this.loneSum(1, 6, 0); // 7.

		this.loneSum(1, 6, 1); // 6.

		this.loneSum(1, 6, 2); // 9.

		this.loneSum(1, 6, 3); // 10.

		this.loneSum(1, 6, 4); // 11.

		this.loneSum(1, 6, 5); // 12.

		this.loneSum(1, 6, 6); // 1.

		this.loneSum(1, 6, 7); // 14.

		this.loneSum(1, 6, 8); // 15.

		this.loneSum(1, 6, 9); // 16.

		this.loneSum(1, 7, 0); // 8.

		this.loneSum(1, 7, 1); // 7.

		this.loneSum(1, 7, 2); // 10.

		this.loneSum(1, 7, 3); // 11.

		this.loneSum(1, 7, 4); // 12.

		this.loneSum(1, 7, 5); // 13.

		this.loneSum(1, 7, 6); // 14.

		this.loneSum(1, 7, 7); // 1.

		this.loneSum(1, 7, 8); // 16.

		this.loneSum(1, 7, 9); // 17.

		this.loneSum(1, 8, 0); // 9.

		this.loneSum(1, 8, 1); // 8.

		this.loneSum(1, 8, 2); // 11.

		this.loneSum(1, 8, 3); // 12.

		this.loneSum(1, 8, 4); // 13.

		this.loneSum(1, 8, 5); // 14.

		this.loneSum(1, 8, 6); // 15.

		this.loneSum(1, 8, 7); // 16.

		this.loneSum(1, 8, 8); // 1.

		this.loneSum(1, 8, 9); // 18.

		this.loneSum(1, 9, 0); // 10.

		this.loneSum(1, 9, 1); // 9.

		this.loneSum(1, 9, 2); // 12.

		this.loneSum(1, 9, 3); // 13.

		this.loneSum(1, 9, 4); // 14.

		this.loneSum(1, 9, 5); // 15.

		this.loneSum(1, 9, 6); // 16.

		this.loneSum(1, 9, 7); // 17.

		this.loneSum(1, 9, 8); // 18.

		this.loneSum(1, 9, 9); // 1.

		this.loneSum(2, 0, 0); // 2.

		this.loneSum(2, 0, 1); // 3.

		this.loneSum(2, 0, 2); // 0.

		this.loneSum(2, 0, 3); // 5.

		this.loneSum(2, 0, 4); // 6.

		this.loneSum(2, 0, 5); // 7.

		this.loneSum(2, 0, 6); // 8.

		this.loneSum(2, 0, 7); // 9.

		this.loneSum(2, 0, 8); // 10.

		this.loneSum(2, 0, 9); // 11.

		this.loneSum(2, 1, 0); // 3.

		this.loneSum(2, 1, 1); // 2.

		this.loneSum(2, 1, 2); // 1.

		this.loneSum(2, 1, 3); // 6.

		this.loneSum(2, 1, 4); // 7.

		this.loneSum(2, 1, 5); // 8.

		this.loneSum(2, 1, 6); // 9.

		this.loneSum(2, 1, 7); // 10.

		this.loneSum(2, 1, 8); // 11.

		this.loneSum(2, 1, 9); // 12.

		this.loneSum(2, 2, 0); // 0.

		this.loneSum(2, 2, 1); // 1.

		this.loneSum(2, 2, 2); // 0.

		this.loneSum(2, 2, 3); // 3.

		this.loneSum(2, 2, 4); // 4.

		this.loneSum(2, 2, 5); // 5.

		this.loneSum(2, 2, 6); // 6.

		this.loneSum(2, 2, 7); // 7.

		this.loneSum(2, 2, 8); // 8.

		this.loneSum(2, 2, 9); // 9.

		this.loneSum(2, 3, 0); // 5.

		this.loneSum(2, 3, 1); // 6.

		this.loneSum(2, 3, 2); // 3.

		this.loneSum(2, 3, 3); // 2.

		this.loneSum(2, 3, 4); // 9.

		this.loneSum(2, 3, 5); // 10.

		this.loneSum(2, 3, 6); // 11.

		this.loneSum(2, 3, 7); // 12.

		this.loneSum(2, 3, 8); // 13.

		this.loneSum(2, 3, 9); // 14.

		this.loneSum(2, 4, 0); // 6.

		this.loneSum(2, 4, 1); // 7.

		this.loneSum(2, 4, 2); // 4.

		this.loneSum(2, 4, 3); // 9.

		this.loneSum(2, 4, 4); // 2.

		this.loneSum(2, 4, 5); // 11.

		this.loneSum(2, 4, 6); // 12.

		this.loneSum(2, 4, 7); // 13.

		this.loneSum(2, 4, 8); // 14.

		this.loneSum(2, 4, 9); // 15.

		this.loneSum(2, 5, 0); // 7.

		this.loneSum(2, 5, 1); // 8.

		this.loneSum(2, 5, 2); // 5.

		this.loneSum(2, 5, 3); // 10.

		this.loneSum(2, 5, 4); // 11.

		this.loneSum(2, 5, 5); // 2.

		this.loneSum(2, 5, 6); // 13.

		this.loneSum(2, 5, 7); // 14.

		this.loneSum(2, 5, 8); // 15.

		this.loneSum(2, 5, 9); // 16.

		this.loneSum(2, 6, 0); // 8.

		this.loneSum(2, 6, 1); // 9.

		this.loneSum(2, 6, 2); // 6.

		this.loneSum(2, 6, 3); // 11.

		this.loneSum(2, 6, 4); // 12.

		this.loneSum(2, 6, 5); // 13.

		this.loneSum(2, 6, 6); // 2.

		this.loneSum(2, 6, 7); // 15.

		this.loneSum(2, 6, 8); // 16.

		this.loneSum(2, 6, 9); // 17.

		this.loneSum(2, 7, 0); // 9.

		this.loneSum(2, 7, 1); // 10.

		this.loneSum(2, 7, 2); // 7.

		this.loneSum(2, 7, 3); // 12.

		this.loneSum(2, 7, 4); // 13.

		this.loneSum(2, 7, 5); // 14.

		this.loneSum(2, 7, 6); // 15.

		this.loneSum(2, 7, 7); // 2.

		this.loneSum(2, 7, 8); // 17.

		this.loneSum(2, 7, 9); // 18.

		this.loneSum(2, 8, 0); // 10.

		this.loneSum(2, 8, 1); // 11.

		this.loneSum(2, 8, 2); // 8.

		this.loneSum(2, 8, 3); // 13.

		this.loneSum(2, 8, 4); // 14.

		this.loneSum(2, 8, 5); // 15.

		this.loneSum(2, 8, 6); // 16.

		this.loneSum(2, 8, 7); // 17.

		this.loneSum(2, 8, 8); // 2.

		this.loneSum(2, 8, 9); // 19.

		this.loneSum(2, 9, 0); // 11.

		this.loneSum(2, 9, 1); // 12.

		this.loneSum(2, 9, 2); // 9.

		this.loneSum(2, 9, 3); // 14.

		this.loneSum(2, 9, 4); // 15.

		this.loneSum(2, 9, 5); // 16.

		this.loneSum(2, 9, 6); // 17.

		this.loneSum(2, 9, 7); // 18.

		this.loneSum(2, 9, 8); // 19.

		this.loneSum(2, 9, 9); // 2.

		this.loneSum(3, 0, 0); // 3.

		this.loneSum(3, 0, 1); // 4.

		this.loneSum(3, 0, 2); // 5.

		this.loneSum(3, 0, 3); // 0.

		this.loneSum(3, 0, 4); // 7.

		this.loneSum(3, 0, 5); // 8.

		this.loneSum(3, 0, 6); // 9.

		this.loneSum(3, 0, 7); // 10.

		this.loneSum(3, 0, 8); // 11.

		this.loneSum(3, 0, 9); // 12.

		this.loneSum(3, 1, 0); // 4.

		this.loneSum(3, 1, 1); // 3.

		this.loneSum(3, 1, 2); // 6.

		this.loneSum(3, 1, 3); // 1.

		this.loneSum(3, 1, 4); // 8.

		this.loneSum(3, 1, 5); // 9.

		this.loneSum(3, 1, 6); // 10.

		this.loneSum(3, 1, 7); // 11.

		this.loneSum(3, 1, 8); // 12.

		this.loneSum(3, 1, 9); // 13.

		this.loneSum(3, 2, 0); // 5.

		this.loneSum(3, 2, 1); // 6.

		this.loneSum(3, 2, 2); // 3.

		this.loneSum(3, 2, 3); // 2.

		this.loneSum(3, 2, 4); // 9.

		this.loneSum(3, 2, 5); // 10.

		this.loneSum(3, 2, 6); // 11.

		this.loneSum(3, 2, 7); // 12.

		this.loneSum(3, 2, 8); // 13.

		this.loneSum(3, 2, 9); // 14.

		this.loneSum(3, 3, 0); // 0.

		this.loneSum(3, 3, 1); // 1.

		this.loneSum(3, 3, 2); // 2.

		this.loneSum(3, 3, 3); // 0.

		this.loneSum(3, 3, 4); // 4.

		this.loneSum(3, 3, 5); // 5.

		this.loneSum(3, 3, 6); // 6.

		this.loneSum(3, 3, 7); // 7.

		this.loneSum(3, 3, 8); // 8.

		this.loneSum(3, 3, 9); // 9.

		this.loneSum(3, 4, 0); // 7.

		this.loneSum(3, 4, 1); // 8.

		this.loneSum(3, 4, 2); // 9.

		this.loneSum(3, 4, 3); // 4.

		this.loneSum(3, 4, 4); // 3.

		this.loneSum(3, 4, 5); // 12.

		this.loneSum(3, 4, 6); // 13.

		this.loneSum(3, 4, 7); // 14.

		this.loneSum(3, 4, 8); // 15.

		this.loneSum(3, 4, 9); // 16.

		this.loneSum(3, 5, 0); // 8.

		this.loneSum(3, 5, 1); // 9.

		this.loneSum(3, 5, 2); // 10.

		this.loneSum(3, 5, 3); // 5.

		this.loneSum(3, 5, 4); // 12.

		this.loneSum(3, 5, 5); // 3.

		this.loneSum(3, 5, 6); // 14.

		this.loneSum(3, 5, 7); // 15.

		this.loneSum(3, 5, 8); // 16.

		this.loneSum(3, 5, 9); // 17.

		this.loneSum(3, 6, 0); // 9.

		this.loneSum(3, 6, 1); // 10.

		this.loneSum(3, 6, 2); // 11.

		this.loneSum(3, 6, 3); // 6.

		this.loneSum(3, 6, 4); // 13.

		this.loneSum(3, 6, 5); // 14.

		this.loneSum(3, 6, 6); // 3.

		this.loneSum(3, 6, 7); // 16.

		this.loneSum(3, 6, 8); // 17.

		this.loneSum(3, 6, 9); // 18.

		this.loneSum(3, 7, 0); // 10.

		this.loneSum(3, 7, 1); // 11.

		this.loneSum(3, 7, 2); // 12.

		this.loneSum(3, 7, 3); // 7.

		this.loneSum(3, 7, 4); // 14.

		this.loneSum(3, 7, 5); // 15.

		this.loneSum(3, 7, 6); // 16.

		this.loneSum(3, 7, 7); // 3.

		this.loneSum(3, 7, 8); // 18.

		this.loneSum(3, 7, 9); // 19.

		this.loneSum(3, 8, 0); // 11.

		this.loneSum(3, 8, 1); // 12.

		this.loneSum(3, 8, 2); // 13.

		this.loneSum(3, 8, 3); // 8.

		this.loneSum(3, 8, 4); // 15.

		this.loneSum(3, 8, 5); // 16.

		this.loneSum(3, 8, 6); // 17.

		this.loneSum(3, 8, 7); // 18.

		this.loneSum(3, 8, 8); // 3.

		this.loneSum(3, 8, 9); // 20.

		this.loneSum(3, 9, 0); // 12.

		this.loneSum(3, 9, 1); // 13.

		this.loneSum(3, 9, 2); // 14.

		this.loneSum(3, 9, 3); // 9.

		this.loneSum(3, 9, 4); // 16.

		this.loneSum(3, 9, 5); // 17.

		this.loneSum(3, 9, 6); // 18.

		this.loneSum(3, 9, 7); // 19.

		this.loneSum(3, 9, 8); // 20.

		this.loneSum(3, 9, 9); // 3.

		this.loneSum(4, 0, 0); // 4.

		this.loneSum(4, 0, 1); // 5.

		this.loneSum(4, 0, 2); // 6.

		this.loneSum(4, 0, 3); // 7.

		this.loneSum(4, 0, 4); // 0.

		this.loneSum(4, 0, 5); // 9.

		this.loneSum(4, 0, 6); // 10.

		this.loneSum(4, 0, 7); // 11.

		this.loneSum(4, 0, 8); // 12.

		this.loneSum(4, 0, 9); // 13.

		this.loneSum(4, 1, 0); // 5.

		this.loneSum(4, 1, 1); // 4.

		this.loneSum(4, 1, 2); // 7.

		this.loneSum(4, 1, 3); // 8.

		this.loneSum(4, 1, 4); // 1.

		this.loneSum(4, 1, 5); // 10.

		this.loneSum(4, 1, 6); // 11.

		this.loneSum(4, 1, 7); // 12.

		this.loneSum(4, 1, 8); // 13.

		this.loneSum(4, 1, 9); // 14.

		this.loneSum(4, 2, 0); // 6.

		this.loneSum(4, 2, 1); // 7.

		this.loneSum(4, 2, 2); // 4.

		this.loneSum(4, 2, 3); // 9.

		this.loneSum(4, 2, 4); // 2.

		this.loneSum(4, 2, 5); // 11.

		this.loneSum(4, 2, 6); // 12.

		this.loneSum(4, 2, 7); // 13.

		this.loneSum(4, 2, 8); // 14.

		this.loneSum(4, 2, 9); // 15.

		this.loneSum(4, 3, 0); // 7.

		this.loneSum(4, 3, 1); // 8.

		this.loneSum(4, 3, 2); // 9.

		this.loneSum(4, 3, 3); // 4.

		this.loneSum(4, 3, 4); // 3.

		this.loneSum(4, 3, 5); // 12.

		this.loneSum(4, 3, 6); // 13.

		this.loneSum(4, 3, 7); // 14.

		this.loneSum(4, 3, 8); // 15.

		this.loneSum(4, 3, 9); // 16.

		this.loneSum(4, 4, 0); // 0.

		this.loneSum(4, 4, 1); // 1.

		this.loneSum(4, 4, 2); // 2.

		this.loneSum(4, 4, 3); // 3.

		this.loneSum(4, 4, 4); // 0.

		this.loneSum(4, 4, 5); // 5.

		this.loneSum(4, 4, 6); // 6.

		this.loneSum(4, 4, 7); // 7.

		this.loneSum(4, 4, 8); // 8.

		this.loneSum(4, 4, 9); // 9.

		this.loneSum(4, 5, 0); // 9.

		this.loneSum(4, 5, 1); // 10.

		this.loneSum(4, 5, 2); // 11.

		this.loneSum(4, 5, 3); // 12.

		this.loneSum(4, 5, 4); // 5.

		this.loneSum(4, 5, 5); // 4.

		this.loneSum(4, 5, 6); // 15.

		this.loneSum(4, 5, 7); // 16.

		this.loneSum(4, 5, 8); // 17.

		this.loneSum(4, 5, 9); // 18.

		this.loneSum(4, 6, 0); // 10.

		this.loneSum(4, 6, 1); // 11.

		this.loneSum(4, 6, 2); // 12.

		this.loneSum(4, 6, 3); // 13.

		this.loneSum(4, 6, 4); // 6.

		this.loneSum(4, 6, 5); // 15.

		this.loneSum(4, 6, 6); // 4.

		this.loneSum(4, 6, 7); // 17.

		this.loneSum(4, 6, 8); // 18.

		this.loneSum(4, 6, 9); // 19.

		this.loneSum(4, 7, 0); // 11.

		this.loneSum(4, 7, 1); // 12.

		this.loneSum(4, 7, 2); // 13.

		this.loneSum(4, 7, 3); // 14.

		this.loneSum(4, 7, 4); // 7.

		this.loneSum(4, 7, 5); // 16.

		this.loneSum(4, 7, 6); // 17.

		this.loneSum(4, 7, 7); // 4.

		this.loneSum(4, 7, 8); // 19.

		this.loneSum(4, 7, 9); // 20.

		this.loneSum(4, 8, 0); // 12.

		this.loneSum(4, 8, 1); // 13.

		this.loneSum(4, 8, 2); // 14.

		this.loneSum(4, 8, 3); // 15.

		this.loneSum(4, 8, 4); // 8.

		this.loneSum(4, 8, 5); // 17.

		this.loneSum(4, 8, 6); // 18.

		this.loneSum(4, 8, 7); // 19.

		this.loneSum(4, 8, 8); // 4.

		this.loneSum(4, 8, 9); // 21.

		this.loneSum(4, 9, 0); // 13.

		this.loneSum(4, 9, 1); // 14.

		this.loneSum(4, 9, 2); // 15.

		this.loneSum(4, 9, 3); // 16.

		this.loneSum(4, 9, 4); // 9.

		this.loneSum(4, 9, 5); // 18.

		this.loneSum(4, 9, 6); // 19.

		this.loneSum(4, 9, 7); // 20.

		this.loneSum(4, 9, 8); // 21.

		this.loneSum(4, 9, 9); // 4.

		this.loneSum(5, 0, 0); // 5.

		this.loneSum(5, 0, 1); // 6.

		this.loneSum(5, 0, 2); // 7.

		this.loneSum(5, 0, 3); // 8.

		this.loneSum(5, 0, 4); // 9.

		this.loneSum(5, 0, 5); // 0.

		this.loneSum(5, 0, 6); // 11.

		this.loneSum(5, 0, 7); // 12.

		this.loneSum(5, 0, 8); // 13.

		this.loneSum(5, 0, 9); // 14.

		this.loneSum(5, 1, 0); // 6.

		this.loneSum(5, 1, 1); // 5.

		this.loneSum(5, 1, 2); // 8.

		this.loneSum(5, 1, 3); // 9.

		this.loneSum(5, 1, 4); // 10.

		this.loneSum(5, 1, 5); // 1.

		this.loneSum(5, 1, 6); // 12.

		this.loneSum(5, 1, 7); // 13.

		this.loneSum(5, 1, 8); // 14.

		this.loneSum(5, 1, 9); // 15.

		this.loneSum(5, 2, 0); // 7.

		this.loneSum(5, 2, 1); // 8.

		this.loneSum(5, 2, 2); // 5.

		this.loneSum(5, 2, 3); // 10.

		this.loneSum(5, 2, 4); // 11.

		this.loneSum(5, 2, 5); // 2.

		this.loneSum(5, 2, 6); // 13.

		this.loneSum(5, 2, 7); // 14.

		this.loneSum(5, 2, 8); // 15.

		this.loneSum(5, 2, 9); // 16.

		this.loneSum(5, 3, 0); // 8.

		this.loneSum(5, 3, 1); // 9.

		this.loneSum(5, 3, 2); // 10.

		this.loneSum(5, 3, 3); // 5.

		this.loneSum(5, 3, 4); // 12.

		this.loneSum(5, 3, 5); // 3.

		this.loneSum(5, 3, 6); // 14.

		this.loneSum(5, 3, 7); // 15.

		this.loneSum(5, 3, 8); // 16.

		this.loneSum(5, 3, 9); // 17.

		this.loneSum(5, 4, 0); // 9.

		this.loneSum(5, 4, 1); // 10.

		this.loneSum(5, 4, 2); // 11.

		this.loneSum(5, 4, 3); // 12.

		this.loneSum(5, 4, 4); // 5.

		this.loneSum(5, 4, 5); // 4.

		this.loneSum(5, 4, 6); // 15.

		this.loneSum(5, 4, 7); // 16.

		this.loneSum(5, 4, 8); // 17.

		this.loneSum(5, 4, 9); // 18.

		this.loneSum(5, 5, 0); // 0.

		this.loneSum(5, 5, 1); // 1.

		this.loneSum(5, 5, 2); // 2.

		this.loneSum(5, 5, 3); // 3.

		this.loneSum(5, 5, 4); // 4.

		this.loneSum(5, 5, 5); // 0.

		this.loneSum(5, 5, 6); // 6.

		this.loneSum(5, 5, 7); // 7.

		this.loneSum(5, 5, 8); // 8.

		this.loneSum(5, 5, 9); // 9.

		this.loneSum(5, 6, 0); // 11.

		this.loneSum(5, 6, 1); // 12.

		this.loneSum(5, 6, 2); // 13.

		this.loneSum(5, 6, 3); // 14.

		this.loneSum(5, 6, 4); // 15.

		this.loneSum(5, 6, 5); // 6.

		this.loneSum(5, 6, 6); // 5.

		this.loneSum(5, 6, 7); // 18.

		this.loneSum(5, 6, 8); // 19.

		this.loneSum(5, 6, 9); // 20.

		this.loneSum(5, 7, 0); // 12.

		this.loneSum(5, 7, 1); // 13.

		this.loneSum(5, 7, 2); // 14.

		this.loneSum(5, 7, 3); // 15.

		this.loneSum(5, 7, 4); // 16.

		this.loneSum(5, 7, 5); // 7.

		this.loneSum(5, 7, 6); // 18.

		this.loneSum(5, 7, 7); // 5.

		this.loneSum(5, 7, 8); // 20.

		this.loneSum(5, 7, 9); // 21.

		this.loneSum(5, 8, 0); // 13.

		this.loneSum(5, 8, 1); // 14.

		this.loneSum(5, 8, 2); // 15.

		this.loneSum(5, 8, 3); // 16.

		this.loneSum(5, 8, 4); // 17.

		this.loneSum(5, 8, 5); // 8.

		this.loneSum(5, 8, 6); // 19.

		this.loneSum(5, 8, 7); // 20.

		this.loneSum(5, 8, 8); // 5.

		this.loneSum(5, 8, 9); // 22.

		this.loneSum(5, 9, 0); // 14.

		this.loneSum(5, 9, 1); // 15.

		this.loneSum(5, 9, 2); // 16.

		this.loneSum(5, 9, 3); // 17.

		this.loneSum(5, 9, 4); // 18.

		this.loneSum(5, 9, 5); // 9.

		this.loneSum(5, 9, 6); // 20.

		this.loneSum(5, 9, 7); // 21.

		this.loneSum(5, 9, 8); // 22.

		this.loneSum(5, 9, 9); // 5.

		this.loneSum(6, 0, 0); // 6.

		this.loneSum(6, 0, 1); // 7.

		this.loneSum(6, 0, 2); // 8.

		this.loneSum(6, 0, 3); // 9.

		this.loneSum(6, 0, 4); // 10.

		this.loneSum(6, 0, 5); // 11.

		this.loneSum(6, 0, 6); // 0.

		this.loneSum(6, 0, 7); // 13.

		this.loneSum(6, 0, 8); // 14.

		this.loneSum(6, 0, 9); // 15.

		this.loneSum(6, 1, 0); // 7.

		this.loneSum(6, 1, 1); // 6.

		this.loneSum(6, 1, 2); // 9.

		this.loneSum(6, 1, 3); // 10.

		this.loneSum(6, 1, 4); // 11.

		this.loneSum(6, 1, 5); // 12.

		this.loneSum(6, 1, 6); // 1.

		this.loneSum(6, 1, 7); // 14.

		this.loneSum(6, 1, 8); // 15.

		this.loneSum(6, 1, 9); // 16.

		this.loneSum(6, 2, 0); // 8.

		this.loneSum(6, 2, 1); // 9.

		this.loneSum(6, 2, 2); // 6.

		this.loneSum(6, 2, 3); // 11.

		this.loneSum(6, 2, 4); // 12.

		this.loneSum(6, 2, 5); // 13.

		this.loneSum(6, 2, 6); // 2.

		this.loneSum(6, 2, 7); // 15.

		this.loneSum(6, 2, 8); // 16.

		this.loneSum(6, 2, 9); // 17.

		this.loneSum(6, 3, 0); // 9.

		this.loneSum(6, 3, 1); // 10.

		this.loneSum(6, 3, 2); // 11.

		this.loneSum(6, 3, 3); // 6.

		this.loneSum(6, 3, 4); // 13.

		this.loneSum(6, 3, 5); // 14.

		this.loneSum(6, 3, 6); // 3.

		this.loneSum(6, 3, 7); // 16.

		this.loneSum(6, 3, 8); // 17.

		this.loneSum(6, 3, 9); // 18.

		this.loneSum(6, 4, 0); // 10.

		this.loneSum(6, 4, 1); // 11.

		this.loneSum(6, 4, 2); // 12.

		this.loneSum(6, 4, 3); // 13.

		this.loneSum(6, 4, 4); // 6.

		this.loneSum(6, 4, 5); // 15.

		this.loneSum(6, 4, 6); // 4.

		this.loneSum(6, 4, 7); // 17.

		this.loneSum(6, 4, 8); // 18.

		this.loneSum(6, 4, 9); // 19.

		this.loneSum(6, 5, 0); // 11.

		this.loneSum(6, 5, 1); // 12.

		this.loneSum(6, 5, 2); // 13.

		this.loneSum(6, 5, 3); // 14.

		this.loneSum(6, 5, 4); // 15.

		this.loneSum(6, 5, 5); // 6.

		this.loneSum(6, 5, 6); // 5.

		this.loneSum(6, 5, 7); // 18.

		this.loneSum(6, 5, 8); // 19.

		this.loneSum(6, 5, 9); // 20.

		this.loneSum(6, 6, 0); // 0.

		this.loneSum(6, 6, 1); // 1.

		this.loneSum(6, 6, 2); // 2.

		this.loneSum(6, 6, 3); // 3.

		this.loneSum(6, 6, 4); // 4.

		this.loneSum(6, 6, 5); // 5.

		this.loneSum(6, 6, 6); // 0.

		this.loneSum(6, 6, 7); // 7.

		this.loneSum(6, 6, 8); // 8.

		this.loneSum(6, 6, 9); // 9.

		this.loneSum(6, 7, 0); // 13.

		this.loneSum(6, 7, 1); // 14.

		this.loneSum(6, 7, 2); // 15.

		this.loneSum(6, 7, 3); // 16.

		this.loneSum(6, 7, 4); // 17.

		this.loneSum(6, 7, 5); // 18.

		this.loneSum(6, 7, 6); // 7.

		this.loneSum(6, 7, 7); // 6.

		this.loneSum(6, 7, 8); // 21.

		this.loneSum(6, 7, 9); // 22.

		this.loneSum(6, 8, 0); // 14.

		this.loneSum(6, 8, 1); // 15.

		this.loneSum(6, 8, 2); // 16.

		this.loneSum(6, 8, 3); // 17.

		this.loneSum(6, 8, 4); // 18.

		this.loneSum(6, 8, 5); // 19.

		this.loneSum(6, 8, 6); // 8.

		this.loneSum(6, 8, 7); // 21.

		this.loneSum(6, 8, 8); // 6.

		this.loneSum(6, 8, 9); // 23.

		this.loneSum(6, 9, 0); // 15.

		this.loneSum(6, 9, 1); // 16.

		this.loneSum(6, 9, 2); // 17.

		this.loneSum(6, 9, 3); // 18.

		this.loneSum(6, 9, 4); // 19.

		this.loneSum(6, 9, 5); // 20.

		this.loneSum(6, 9, 6); // 9.

		this.loneSum(6, 9, 7); // 22.

		this.loneSum(6, 9, 8); // 23.

		this.loneSum(6, 9, 9); // 6.

		this.loneSum(7, 0, 0); // 7.

		this.loneSum(7, 0, 1); // 8.

		this.loneSum(7, 0, 2); // 9.

		this.loneSum(7, 0, 3); // 10.

		this.loneSum(7, 0, 4); // 11.

		this.loneSum(7, 0, 5); // 12.

		this.loneSum(7, 0, 6); // 13.

		this.loneSum(7, 0, 7); // 0.

		this.loneSum(7, 0, 8); // 15.

		this.loneSum(7, 0, 9); // 16.

		this.loneSum(7, 1, 0); // 8.

		this.loneSum(7, 1, 1); // 7.

		this.loneSum(7, 1, 2); // 10.

		this.loneSum(7, 1, 3); // 11.

		this.loneSum(7, 1, 4); // 12.

		this.loneSum(7, 1, 5); // 13.

		this.loneSum(7, 1, 6); // 14.

		this.loneSum(7, 1, 7); // 1.

		this.loneSum(7, 1, 8); // 16.

		this.loneSum(7, 1, 9); // 17.

		this.loneSum(7, 2, 0); // 9.

		this.loneSum(7, 2, 1); // 10.

		this.loneSum(7, 2, 2); // 7.

		this.loneSum(7, 2, 3); // 12.

		this.loneSum(7, 2, 4); // 13.

		this.loneSum(7, 2, 5); // 14.

		this.loneSum(7, 2, 6); // 15.

		this.loneSum(7, 2, 7); // 2.

		this.loneSum(7, 2, 8); // 17.

		this.loneSum(7, 2, 9); // 18.

		this.loneSum(7, 3, 0); // 10.

		this.loneSum(7, 3, 1); // 11.

		this.loneSum(7, 3, 2); // 12.

		this.loneSum(7, 3, 3); // 7.

		this.loneSum(7, 3, 4); // 14.

		this.loneSum(7, 3, 5); // 15.

		this.loneSum(7, 3, 6); // 16.

		this.loneSum(7, 3, 7); // 3.

		this.loneSum(7, 3, 8); // 18.

		this.loneSum(7, 3, 9); // 19.

		this.loneSum(7, 4, 0); // 11.

		this.loneSum(7, 4, 1); // 12.

		this.loneSum(7, 4, 2); // 13.

		this.loneSum(7, 4, 3); // 14.

		this.loneSum(7, 4, 4); // 7.

		this.loneSum(7, 4, 5); // 16.

		this.loneSum(7, 4, 6); // 17.

		this.loneSum(7, 4, 7); // 4.

		this.loneSum(7, 4, 8); // 19.

		this.loneSum(7, 4, 9); // 20.

		this.loneSum(7, 5, 0); // 12.

		this.loneSum(7, 5, 1); // 13.

		this.loneSum(7, 5, 2); // 14.

		this.loneSum(7, 5, 3); // 15.

		this.loneSum(7, 5, 4); // 16.

		this.loneSum(7, 5, 5); // 7.

		this.loneSum(7, 5, 6); // 18.

		this.loneSum(7, 5, 7); // 5.

		this.loneSum(7, 5, 8); // 20.

		this.loneSum(7, 5, 9); // 21.

		this.loneSum(7, 6, 0); // 13.

		this.loneSum(7, 6, 1); // 14.

		this.loneSum(7, 6, 2); // 15.

		this.loneSum(7, 6, 3); // 16.

		this.loneSum(7, 6, 4); // 17.

		this.loneSum(7, 6, 5); // 18.

		this.loneSum(7, 6, 6); // 7.

		this.loneSum(7, 6, 7); // 6.

		this.loneSum(7, 6, 8); // 21.

		this.loneSum(7, 6, 9); // 22.

		this.loneSum(7, 7, 0); // 0.

		this.loneSum(7, 7, 1); // 1.

		this.loneSum(7, 7, 2); // 2.

		this.loneSum(7, 7, 3); // 3.

		this.loneSum(7, 7, 4); // 4.

		this.loneSum(7, 7, 5); // 5.

		this.loneSum(7, 7, 6); // 6.

		this.loneSum(7, 7, 7); // 0.

		this.loneSum(7, 7, 8); // 8.

		this.loneSum(7, 7, 9); // 9.

		this.loneSum(7, 8, 0); // 15.

		this.loneSum(7, 8, 1); // 16.

		this.loneSum(7, 8, 2); // 17.

		this.loneSum(7, 8, 3); // 18.

		this.loneSum(7, 8, 4); // 19.

		this.loneSum(7, 8, 5); // 20.

		this.loneSum(7, 8, 6); // 21.

		this.loneSum(7, 8, 7); // 8.

		this.loneSum(7, 8, 8); // 7.

		this.loneSum(7, 8, 9); // 24.

		this.loneSum(7, 9, 0); // 16.

		this.loneSum(7, 9, 1); // 17.

		this.loneSum(7, 9, 2); // 18.

		this.loneSum(7, 9, 3); // 19.

		this.loneSum(7, 9, 4); // 20.

		this.loneSum(7, 9, 5); // 21.

		this.loneSum(7, 9, 6); // 22.

		this.loneSum(7, 9, 7); // 9.

		this.loneSum(7, 9, 8); // 24.

		this.loneSum(7, 9, 9); // 7.

		this.loneSum(8, 0, 0); // 8.

		this.loneSum(8, 0, 1); // 9.

		this.loneSum(8, 0, 2); // 10.

		this.loneSum(8, 0, 3); // 11.

		this.loneSum(8, 0, 4); // 12.

		this.loneSum(8, 0, 5); // 13.

		this.loneSum(8, 0, 6); // 14.

		this.loneSum(8, 0, 7); // 15.

		this.loneSum(8, 0, 8); // 0.

		this.loneSum(8, 0, 9); // 17.

		this.loneSum(8, 1, 0); // 9.

		this.loneSum(8, 1, 1); // 8.

		this.loneSum(8, 1, 2); // 11.

		this.loneSum(8, 1, 3); // 12.

		this.loneSum(8, 1, 4); // 13.

		this.loneSum(8, 1, 5); // 14.

		this.loneSum(8, 1, 6); // 15.

		this.loneSum(8, 1, 7); // 16.

		this.loneSum(8, 1, 8); // 1.

		this.loneSum(8, 1, 9); // 18.

		this.loneSum(8, 2, 0); // 10.

		this.loneSum(8, 2, 1); // 11.

		this.loneSum(8, 2, 2); // 8.

		this.loneSum(8, 2, 3); // 13.

		this.loneSum(8, 2, 4); // 14.

		this.loneSum(8, 2, 5); // 15.

		this.loneSum(8, 2, 6); // 16.

		this.loneSum(8, 2, 7); // 17.

		this.loneSum(8, 2, 8); // 2.

		this.loneSum(8, 2, 9); // 19.

		this.loneSum(8, 3, 0); // 11.

		this.loneSum(8, 3, 1); // 12.

		this.loneSum(8, 3, 2); // 13.

		this.loneSum(8, 3, 3); // 8.

		this.loneSum(8, 3, 4); // 15.

		this.loneSum(8, 3, 5); // 16.

		this.loneSum(8, 3, 6); // 17.

		this.loneSum(8, 3, 7); // 18.

		this.loneSum(8, 3, 8); // 3.

		this.loneSum(8, 3, 9); // 20.

		this.loneSum(8, 4, 0); // 12.

		this.loneSum(8, 4, 1); // 13.

		this.loneSum(8, 4, 2); // 14.

		this.loneSum(8, 4, 3); // 15.

		this.loneSum(8, 4, 4); // 8.

		this.loneSum(8, 4, 5); // 17.

		this.loneSum(8, 4, 6); // 18.

		this.loneSum(8, 4, 7); // 19.

		this.loneSum(8, 4, 8); // 4.

		this.loneSum(8, 4, 9); // 21.

		this.loneSum(8, 5, 0); // 13.

		this.loneSum(8, 5, 1); // 14.

		this.loneSum(8, 5, 2); // 15.

		this.loneSum(8, 5, 3); // 16.

		this.loneSum(8, 5, 4); // 17.

		this.loneSum(8, 5, 5); // 8.

		this.loneSum(8, 5, 6); // 19.

		this.loneSum(8, 5, 7); // 20.

		this.loneSum(8, 5, 8); // 5.

		this.loneSum(8, 5, 9); // 22.

		this.loneSum(8, 6, 0); // 14.

		this.loneSum(8, 6, 1); // 15.

		this.loneSum(8, 6, 2); // 16.

		this.loneSum(8, 6, 3); // 17.

		this.loneSum(8, 6, 4); // 18.

		this.loneSum(8, 6, 5); // 19.

		this.loneSum(8, 6, 6); // 8.

		this.loneSum(8, 6, 7); // 21.

		this.loneSum(8, 6, 8); // 6.

		this.loneSum(8, 6, 9); // 23.

		this.loneSum(8, 7, 0); // 15.

		this.loneSum(8, 7, 1); // 16.

		this.loneSum(8, 7, 2); // 17.

		this.loneSum(8, 7, 3); // 18.

		this.loneSum(8, 7, 4); // 19.

		this.loneSum(8, 7, 5); // 20.

		this.loneSum(8, 7, 6); // 21.

		this.loneSum(8, 7, 7); // 8.

		this.loneSum(8, 7, 8); // 7.

		this.loneSum(8, 7, 9); // 24.

		this.loneSum(8, 8, 0); // 0.

		this.loneSum(8, 8, 1); // 1.

		this.loneSum(8, 8, 2); // 2.

		this.loneSum(8, 8, 3); // 3.

		this.loneSum(8, 8, 4); // 4.

		this.loneSum(8, 8, 5); // 5.

		this.loneSum(8, 8, 6); // 6.

		this.loneSum(8, 8, 7); // 7.

		this.loneSum(8, 8, 8); // 0.

		this.loneSum(8, 8, 9); // 9.

		this.loneSum(8, 9, 0); // 17.

		this.loneSum(8, 9, 1); // 18.

		this.loneSum(8, 9, 2); // 19.

		this.loneSum(8, 9, 3); // 20.

		this.loneSum(8, 9, 4); // 21.

		this.loneSum(8, 9, 5); // 22.

		this.loneSum(8, 9, 6); // 23.

		this.loneSum(8, 9, 7); // 24.

		this.loneSum(8, 9, 8); // 9.

		this.loneSum(8, 9, 9); // 8.

		this.loneSum(9, 0, 0); // 9.

		this.loneSum(9, 0, 1); // 10.

		this.loneSum(9, 0, 2); // 11.

		this.loneSum(9, 0, 3); // 12.

		this.loneSum(9, 0, 4); // 13.

		this.loneSum(9, 0, 5); // 14.

		this.loneSum(9, 0, 6); // 15.

		this.loneSum(9, 0, 7); // 16.

		this.loneSum(9, 0, 8); // 17.

		this.loneSum(9, 0, 9); // 0.

		this.loneSum(9, 1, 0); // 10.

		this.loneSum(9, 1, 1); // 9.

		this.loneSum(9, 1, 2); // 12.

		this.loneSum(9, 1, 3); // 13.

		this.loneSum(9, 1, 4); // 14.

		this.loneSum(9, 1, 5); // 15.

		this.loneSum(9, 1, 6); // 16.

		this.loneSum(9, 1, 7); // 17.

		this.loneSum(9, 1, 8); // 18.

		this.loneSum(9, 1, 9); // 1.

		this.loneSum(9, 2, 0); // 11.

		this.loneSum(9, 2, 1); // 12.

		this.loneSum(9, 2, 2); // 9.

		this.loneSum(9, 2, 3); // 14.

		this.loneSum(9, 2, 4); // 15.

		this.loneSum(9, 2, 5); // 16.

		this.loneSum(9, 2, 6); // 17.

		this.loneSum(9, 2, 7); // 18.

		this.loneSum(9, 2, 8); // 19.

		this.loneSum(9, 2, 9); // 2.

		this.loneSum(9, 3, 0); // 12.

		this.loneSum(9, 3, 1); // 13.

		this.loneSum(9, 3, 2); // 14.

		this.loneSum(9, 3, 3); // 9.

		this.loneSum(9, 3, 4); // 16.

		this.loneSum(9, 3, 5); // 17.

		this.loneSum(9, 3, 6); // 18.

		this.loneSum(9, 3, 7); // 19.

		this.loneSum(9, 3, 8); // 20.

		this.loneSum(9, 3, 9); // 3.

		this.loneSum(9, 4, 0); // 13.

		this.loneSum(9, 4, 1); // 14.

		this.loneSum(9, 4, 2); // 15.

		this.loneSum(9, 4, 3); // 16.

		this.loneSum(9, 4, 4); // 9.

		this.loneSum(9, 4, 5); // 18.

		this.loneSum(9, 4, 6); // 19.

		this.loneSum(9, 4, 7); // 20.

		this.loneSum(9, 4, 8); // 21.

		this.loneSum(9, 4, 9); // 4.

		this.loneSum(9, 5, 0); // 14.

		this.loneSum(9, 5, 1); // 15.

		this.loneSum(9, 5, 2); // 16.

		this.loneSum(9, 5, 3); // 17.

		this.loneSum(9, 5, 4); // 18.

		this.loneSum(9, 5, 5); // 9.

		this.loneSum(9, 5, 6); // 20.

		this.loneSum(9, 5, 7); // 21.

		this.loneSum(9, 5, 8); // 22.

		this.loneSum(9, 5, 9); // 5.

		this.loneSum(9, 6, 0); // 15.

		this.loneSum(9, 6, 1); // 16.

		this.loneSum(9, 6, 2); // 17.

		this.loneSum(9, 6, 3); // 18.

		this.loneSum(9, 6, 4); // 19.

		this.loneSum(9, 6, 5); // 20.

		this.loneSum(9, 6, 6); // 9.

		this.loneSum(9, 6, 7); // 22.

		this.loneSum(9, 6, 8); // 23.

		this.loneSum(9, 6, 9); // 6.

		this.loneSum(9, 7, 0); // 16.

		this.loneSum(9, 7, 1); // 17.

		this.loneSum(9, 7, 2); // 18.

		this.loneSum(9, 7, 3); // 19.

		this.loneSum(9, 7, 4); // 20.

		this.loneSum(9, 7, 5); // 21.

		this.loneSum(9, 7, 6); // 22.

		this.loneSum(9, 7, 7); // 9.

		this.loneSum(9, 7, 8); // 24.

		this.loneSum(9, 7, 9); // 7.

		this.loneSum(9, 8, 0); // 17.

		this.loneSum(9, 8, 1); // 18.

		this.loneSum(9, 8, 2); // 19.

		this.loneSum(9, 8, 3); // 20.

		this.loneSum(9, 8, 4); // 21.

		this.loneSum(9, 8, 5); // 22.

		this.loneSum(9, 8, 6); // 23.

		this.loneSum(9, 8, 7); // 24.

		this.loneSum(9, 8, 8); // 9.

		this.loneSum(9, 8, 9); // 8.

		this.loneSum(9, 9, 0); // 0.

		this.loneSum(9, 9, 1); // 1.

		this.loneSum(9, 9, 2); // 2.

		this.loneSum(9, 9, 3); // 3.

		this.loneSum(9, 9, 4); // 4.

		this.loneSum(9, 9, 5); // 5.

		this.loneSum(9, 9, 6); // 6.

		this.loneSum(9, 9, 7); // 7.

		this.loneSum(9, 9, 8); // 8.

		this.loneSum(9, 9, 9); // 0.



		// luckySum(a, b, c) Test Cases



		this.luckySum(0, 0, 0); // 0.

		this.luckySum(0, 0, 1); // 1.

		this.luckySum(0, 0, 2); // 2.

		this.luckySum(0, 0, 3); // 3.

		this.luckySum(0, 0, 4); // 4.

		this.luckySum(0, 0, 5); // 5.

		this.luckySum(0, 0, 6); // 6.

		this.luckySum(0, 0, 7); // 7.

		this.luckySum(0, 0, 8); // 8.

		this.luckySum(0, 0, 9); // 9.

		this.luckySum(0, 1, 0); // 1.

		this.luckySum(0, 1, 1); // 2.

		this.luckySum(0, 1, 2); // 3.

		this.luckySum(0, 1, 3); // 4.

		this.luckySum(0, 1, 4); // 5.

		this.luckySum(0, 1, 5); // 6.

		this.luckySum(0, 1, 6); // 7.

		this.luckySum(0, 1, 7); // 8.

		this.luckySum(0, 1, 8); // 9.

		this.luckySum(0, 1, 9); // 10.

		this.luckySum(0, 2, 0); // 2.

		this.luckySum(0, 2, 1); // 3.

		this.luckySum(0, 2, 2); // 4.

		this.luckySum(0, 2, 3); // 5.

		this.luckySum(0, 2, 4); // 6.

		this.luckySum(0, 2, 5); // 7.

		this.luckySum(0, 2, 6); // 8.

		this.luckySum(0, 2, 7); // 9.

		this.luckySum(0, 2, 8); // 10.

		this.luckySum(0, 2, 9); // 11.

		this.luckySum(0, 3, 0); // 3.

		this.luckySum(0, 3, 1); // 4.

		this.luckySum(0, 3, 2); // 5.

		this.luckySum(0, 3, 3); // 6.

		this.luckySum(0, 3, 4); // 7.

		this.luckySum(0, 3, 5); // 8.

		this.luckySum(0, 3, 6); // 9.

		this.luckySum(0, 3, 7); // 10.

		this.luckySum(0, 3, 8); // 11.

		this.luckySum(0, 3, 9); // 12.

		this.luckySum(0, 4, 0); // 4.

		this.luckySum(0, 4, 1); // 5.

		this.luckySum(0, 4, 2); // 6.

		this.luckySum(0, 4, 3); // 7.

		this.luckySum(0, 4, 4); // 8.

		this.luckySum(0, 4, 5); // 9.

		this.luckySum(0, 4, 6); // 10.

		this.luckySum(0, 4, 7); // 11.

		this.luckySum(0, 4, 8); // 12.

		this.luckySum(0, 4, 9); // 13.

		this.luckySum(0, 5, 0); // 5.

		this.luckySum(0, 5, 1); // 6.

		this.luckySum(0, 5, 2); // 7.

		this.luckySum(0, 5, 3); // 8.

		this.luckySum(0, 5, 4); // 9.

		this.luckySum(0, 5, 5); // 10.

		this.luckySum(0, 5, 6); // 11.

		this.luckySum(0, 5, 7); // 12.

		this.luckySum(0, 5, 8); // 13.

		this.luckySum(0, 5, 9); // 14.

		this.luckySum(0, 6, 0); // 6.

		this.luckySum(0, 6, 1); // 7.

		this.luckySum(0, 6, 2); // 8.

		this.luckySum(0, 6, 3); // 9.

		this.luckySum(0, 6, 4); // 10.

		this.luckySum(0, 6, 5); // 11.

		this.luckySum(0, 6, 6); // 12.

		this.luckySum(0, 6, 7); // 13.

		this.luckySum(0, 6, 8); // 14.

		this.luckySum(0, 6, 9); // 15.

		this.luckySum(0, 7, 0); // 7.

		this.luckySum(0, 7, 1); // 8.

		this.luckySum(0, 7, 2); // 9.

		this.luckySum(0, 7, 3); // 10.

		this.luckySum(0, 7, 4); // 11.

		this.luckySum(0, 7, 5); // 12.

		this.luckySum(0, 7, 6); // 13.

		this.luckySum(0, 7, 7); // 14.

		this.luckySum(0, 7, 8); // 15.

		this.luckySum(0, 7, 9); // 16.

		this.luckySum(0, 8, 0); // 8.

		this.luckySum(0, 8, 1); // 9.

		this.luckySum(0, 8, 2); // 10.

		this.luckySum(0, 8, 3); // 11.

		this.luckySum(0, 8, 4); // 12.

		this.luckySum(0, 8, 5); // 13.

		this.luckySum(0, 8, 6); // 14.

		this.luckySum(0, 8, 7); // 15.

		this.luckySum(0, 8, 8); // 16.

		this.luckySum(0, 8, 9); // 17.

		this.luckySum(0, 9, 0); // 9.

		this.luckySum(0, 9, 1); // 10.

		this.luckySum(0, 9, 2); // 11.

		this.luckySum(0, 9, 3); // 12.

		this.luckySum(0, 9, 4); // 13.

		this.luckySum(0, 9, 5); // 14.

		this.luckySum(0, 9, 6); // 15.

		this.luckySum(0, 9, 7); // 16.

		this.luckySum(0, 9, 8); // 17.

		this.luckySum(0, 9, 9); // 18.

		this.luckySum(1, 0, 0); // 1.

		this.luckySum(1, 0, 1); // 2.

		this.luckySum(1, 0, 2); // 3.

		this.luckySum(1, 0, 3); // 4.

		this.luckySum(1, 0, 4); // 5.

		this.luckySum(1, 0, 5); // 6.

		this.luckySum(1, 0, 6); // 7.

		this.luckySum(1, 0, 7); // 8.

		this.luckySum(1, 0, 8); // 9.

		this.luckySum(1, 0, 9); // 10.

		this.luckySum(1, 1, 0); // 2.

		this.luckySum(1, 1, 1); // 3.

		this.luckySum(1, 1, 2); // 4.

		this.luckySum(1, 1, 3); // 5.

		this.luckySum(1, 1, 4); // 6.

		this.luckySum(1, 1, 5); // 7.

		this.luckySum(1, 1, 6); // 8.

		this.luckySum(1, 1, 7); // 9.

		this.luckySum(1, 1, 8); // 10.

		this.luckySum(1, 1, 9); // 11.

		this.luckySum(1, 2, 0); // 3.

		this.luckySum(1, 2, 1); // 4.

		this.luckySum(1, 2, 2); // 5.

		this.luckySum(1, 2, 3); // 6.

		this.luckySum(1, 2, 4); // 7.

		this.luckySum(1, 2, 5); // 8.

		this.luckySum(1, 2, 6); // 9.

		this.luckySum(1, 2, 7); // 10.

		this.luckySum(1, 2, 8); // 11.

		this.luckySum(1, 2, 9); // 12.

		this.luckySum(1, 3, 0); // 4.

		this.luckySum(1, 3, 1); // 5.

		this.luckySum(1, 3, 2); // 6.

		this.luckySum(1, 3, 3); // 7.

		this.luckySum(1, 3, 4); // 8.

		this.luckySum(1, 3, 5); // 9.

		this.luckySum(1, 3, 6); // 10.

		this.luckySum(1, 3, 7); // 11.

		this.luckySum(1, 3, 8); // 12.

		this.luckySum(1, 3, 9); // 13.

		this.luckySum(1, 4, 0); // 5.

		this.luckySum(1, 4, 1); // 6.

		this.luckySum(1, 4, 2); // 7.

		this.luckySum(1, 4, 3); // 8.

		this.luckySum(1, 4, 4); // 9.

		this.luckySum(1, 4, 5); // 10.

		this.luckySum(1, 4, 6); // 11.

		this.luckySum(1, 4, 7); // 12.

		this.luckySum(1, 4, 8); // 13.

		this.luckySum(1, 4, 9); // 14.

		this.luckySum(1, 5, 0); // 6.

		this.luckySum(1, 5, 1); // 7.

		this.luckySum(1, 5, 2); // 8.

		this.luckySum(1, 5, 3); // 9.

		this.luckySum(1, 5, 4); // 10.

		this.luckySum(1, 5, 5); // 11.

		this.luckySum(1, 5, 6); // 12.

		this.luckySum(1, 5, 7); // 13.

		this.luckySum(1, 5, 8); // 14.

		this.luckySum(1, 5, 9); // 15.

		this.luckySum(1, 6, 0); // 7.

		this.luckySum(1, 6, 1); // 8.

		this.luckySum(1, 6, 2); // 9.

		this.luckySum(1, 6, 3); // 10.

		this.luckySum(1, 6, 4); // 11.

		this.luckySum(1, 6, 5); // 12.

		this.luckySum(1, 6, 6); // 13.

		this.luckySum(1, 6, 7); // 14.

		this.luckySum(1, 6, 8); // 15.

		this.luckySum(1, 6, 9); // 16.

		this.luckySum(1, 7, 0); // 8.

		this.luckySum(1, 7, 1); // 9.

		this.luckySum(1, 7, 2); // 10.

		this.luckySum(1, 7, 3); // 11.

		this.luckySum(1, 7, 4); // 12.

		this.luckySum(1, 7, 5); // 13.

		this.luckySum(1, 7, 6); // 14.

		this.luckySum(1, 7, 7); // 15.

		this.luckySum(1, 7, 8); // 16.

		this.luckySum(1, 7, 9); // 17.

		this.luckySum(1, 8, 0); // 9.

		this.luckySum(1, 8, 1); // 10.

		this.luckySum(1, 8, 2); // 11.

		this.luckySum(1, 8, 3); // 12.

		this.luckySum(1, 8, 4); // 13.

		this.luckySum(1, 8, 5); // 14.

		this.luckySum(1, 8, 6); // 15.

		this.luckySum(1, 8, 7); // 16.

		this.luckySum(1, 8, 8); // 17.

		this.luckySum(1, 8, 9); // 18.

		this.luckySum(1, 9, 0); // 10.

		this.luckySum(1, 9, 1); // 11.

		this.luckySum(1, 9, 2); // 12.

		this.luckySum(1, 9, 3); // 13.

		this.luckySum(1, 9, 4); // 14.

		this.luckySum(1, 9, 5); // 15.

		this.luckySum(1, 9, 6); // 16.

		this.luckySum(1, 9, 7); // 17.

		this.luckySum(1, 9, 8); // 18.

		this.luckySum(1, 9, 9); // 19.

		this.luckySum(2, 0, 0); // 2.

		this.luckySum(2, 0, 1); // 3.

		this.luckySum(2, 0, 2); // 4.

		this.luckySum(2, 0, 3); // 5.

		this.luckySum(2, 0, 4); // 6.

		this.luckySum(2, 0, 5); // 7.

		this.luckySum(2, 0, 6); // 8.

		this.luckySum(2, 0, 7); // 9.

		this.luckySum(2, 0, 8); // 10.

		this.luckySum(2, 0, 9); // 11.

		this.luckySum(2, 1, 0); // 3.

		this.luckySum(2, 1, 1); // 4.

		this.luckySum(2, 1, 2); // 5.

		this.luckySum(2, 1, 3); // 6.

		this.luckySum(2, 1, 4); // 7.

		this.luckySum(2, 1, 5); // 8.

		this.luckySum(2, 1, 6); // 9.

		this.luckySum(2, 1, 7); // 10.

		this.luckySum(2, 1, 8); // 11.

		this.luckySum(2, 1, 9); // 12.

		this.luckySum(2, 2, 0); // 4.

		this.luckySum(2, 2, 1); // 5.

		this.luckySum(2, 2, 2); // 6.

		this.luckySum(2, 2, 3); // 7.

		this.luckySum(2, 2, 4); // 8.

		this.luckySum(2, 2, 5); // 9.

		this.luckySum(2, 2, 6); // 10.

		this.luckySum(2, 2, 7); // 11.

		this.luckySum(2, 2, 8); // 12.

		this.luckySum(2, 2, 9); // 13.

		this.luckySum(2, 3, 0); // 5.

		this.luckySum(2, 3, 1); // 6.

		this.luckySum(2, 3, 2); // 7.

		this.luckySum(2, 3, 3); // 8.

		this.luckySum(2, 3, 4); // 9.

		this.luckySum(2, 3, 5); // 10.

		this.luckySum(2, 3, 6); // 11.

		this.luckySum(2, 3, 7); // 12.

		this.luckySum(2, 3, 8); // 13.

		this.luckySum(2, 3, 9); // 14.

		this.luckySum(2, 4, 0); // 6.

		this.luckySum(2, 4, 1); // 7.

		this.luckySum(2, 4, 2); // 8.

		this.luckySum(2, 4, 3); // 9.

		this.luckySum(2, 4, 4); // 10.

		this.luckySum(2, 4, 5); // 11.

		this.luckySum(2, 4, 6); // 12.

		this.luckySum(2, 4, 7); // 13.

		this.luckySum(2, 4, 8); // 14.

		this.luckySum(2, 4, 9); // 15.

		this.luckySum(2, 5, 0); // 7.

		this.luckySum(2, 5, 1); // 8.

		this.luckySum(2, 5, 2); // 9.

		this.luckySum(2, 5, 3); // 10.

		this.luckySum(2, 5, 4); // 11.

		this.luckySum(2, 5, 5); // 12.

		this.luckySum(2, 5, 6); // 13.

		this.luckySum(2, 5, 7); // 14.

		this.luckySum(2, 5, 8); // 15.

		this.luckySum(2, 5, 9); // 16.

		this.luckySum(2, 6, 0); // 8.

		this.luckySum(2, 6, 1); // 9.

		this.luckySum(2, 6, 2); // 10.

		this.luckySum(2, 6, 3); // 11.

		this.luckySum(2, 6, 4); // 12.

		this.luckySum(2, 6, 5); // 13.

		this.luckySum(2, 6, 6); // 14.

		this.luckySum(2, 6, 7); // 15.

		this.luckySum(2, 6, 8); // 16.

		this.luckySum(2, 6, 9); // 17.

		this.luckySum(2, 7, 0); // 9.

		this.luckySum(2, 7, 1); // 10.

		this.luckySum(2, 7, 2); // 11.

		this.luckySum(2, 7, 3); // 12.

		this.luckySum(2, 7, 4); // 13.

		this.luckySum(2, 7, 5); // 14.

		this.luckySum(2, 7, 6); // 15.

		this.luckySum(2, 7, 7); // 16.

		this.luckySum(2, 7, 8); // 17.

		this.luckySum(2, 7, 9); // 18.

		this.luckySum(2, 8, 0); // 10.

		this.luckySum(2, 8, 1); // 11.

		this.luckySum(2, 8, 2); // 12.

		this.luckySum(2, 8, 3); // 13.

		this.luckySum(2, 8, 4); // 14.

		this.luckySum(2, 8, 5); // 15.

		this.luckySum(2, 8, 6); // 16.

		this.luckySum(2, 8, 7); // 17.

		this.luckySum(2, 8, 8); // 18.

		this.luckySum(2, 8, 9); // 19.

		this.luckySum(2, 9, 0); // 11.

		this.luckySum(2, 9, 1); // 12.

		this.luckySum(2, 9, 2); // 13.

		this.luckySum(2, 9, 3); // 14.

		this.luckySum(2, 9, 4); // 15.

		this.luckySum(2, 9, 5); // 16.

		this.luckySum(2, 9, 6); // 17.

		this.luckySum(2, 9, 7); // 18.

		this.luckySum(2, 9, 8); // 19.

		this.luckySum(2, 9, 9); // 20.

		this.luckySum(3, 0, 0); // 3.

		this.luckySum(3, 0, 1); // 4.

		this.luckySum(3, 0, 2); // 5.

		this.luckySum(3, 0, 3); // 6.

		this.luckySum(3, 0, 4); // 7.

		this.luckySum(3, 0, 5); // 8.

		this.luckySum(3, 0, 6); // 9.

		this.luckySum(3, 0, 7); // 10.

		this.luckySum(3, 0, 8); // 11.

		this.luckySum(3, 0, 9); // 12.

		this.luckySum(3, 1, 0); // 4.

		this.luckySum(3, 1, 1); // 5.

		this.luckySum(3, 1, 2); // 6.

		this.luckySum(3, 1, 3); // 7.

		this.luckySum(3, 1, 4); // 8.

		this.luckySum(3, 1, 5); // 9.

		this.luckySum(3, 1, 6); // 10.

		this.luckySum(3, 1, 7); // 11.

		this.luckySum(3, 1, 8); // 12.

		this.luckySum(3, 1, 9); // 13.

		this.luckySum(3, 2, 0); // 5.

		this.luckySum(3, 2, 1); // 6.

		this.luckySum(3, 2, 2); // 7.

		this.luckySum(3, 2, 3); // 8.

		this.luckySum(3, 2, 4); // 9.

		this.luckySum(3, 2, 5); // 10.

		this.luckySum(3, 2, 6); // 11.

		this.luckySum(3, 2, 7); // 12.

		this.luckySum(3, 2, 8); // 13.

		this.luckySum(3, 2, 9); // 14.

		this.luckySum(3, 3, 0); // 6.

		this.luckySum(3, 3, 1); // 7.

		this.luckySum(3, 3, 2); // 8.

		this.luckySum(3, 3, 3); // 9.

		this.luckySum(3, 3, 4); // 10.

		this.luckySum(3, 3, 5); // 11.

		this.luckySum(3, 3, 6); // 12.

		this.luckySum(3, 3, 7); // 13.

		this.luckySum(3, 3, 8); // 14.

		this.luckySum(3, 3, 9); // 15.

		this.luckySum(3, 4, 0); // 7.

		this.luckySum(3, 4, 1); // 8.

		this.luckySum(3, 4, 2); // 9.

		this.luckySum(3, 4, 3); // 10.

		this.luckySum(3, 4, 4); // 11.

		this.luckySum(3, 4, 5); // 12.

		this.luckySum(3, 4, 6); // 13.

		this.luckySum(3, 4, 7); // 14.

		this.luckySum(3, 4, 8); // 15.

		this.luckySum(3, 4, 9); // 16.

		this.luckySum(3, 5, 0); // 8.

		this.luckySum(3, 5, 1); // 9.

		this.luckySum(3, 5, 2); // 10.

		this.luckySum(3, 5, 3); // 11.

		this.luckySum(3, 5, 4); // 12.

		this.luckySum(3, 5, 5); // 13.

		this.luckySum(3, 5, 6); // 14.

		this.luckySum(3, 5, 7); // 15.

		this.luckySum(3, 5, 8); // 16.

		this.luckySum(3, 5, 9); // 17.

		this.luckySum(3, 6, 0); // 9.

		this.luckySum(3, 6, 1); // 10.

		this.luckySum(3, 6, 2); // 11.

		this.luckySum(3, 6, 3); // 12.

		this.luckySum(3, 6, 4); // 13.

		this.luckySum(3, 6, 5); // 14.

		this.luckySum(3, 6, 6); // 15.

		this.luckySum(3, 6, 7); // 16.

		this.luckySum(3, 6, 8); // 17.

		this.luckySum(3, 6, 9); // 18.

		this.luckySum(3, 7, 0); // 10.

		this.luckySum(3, 7, 1); // 11.

		this.luckySum(3, 7, 2); // 12.

		this.luckySum(3, 7, 3); // 13.

		this.luckySum(3, 7, 4); // 14.

		this.luckySum(3, 7, 5); // 15.

		this.luckySum(3, 7, 6); // 16.

		this.luckySum(3, 7, 7); // 17.

		this.luckySum(3, 7, 8); // 18.

		this.luckySum(3, 7, 9); // 19.

		this.luckySum(3, 8, 0); // 11.

		this.luckySum(3, 8, 1); // 12.

		this.luckySum(3, 8, 2); // 13.

		this.luckySum(3, 8, 3); // 14.

		this.luckySum(3, 8, 4); // 15.

		this.luckySum(3, 8, 5); // 16.

		this.luckySum(3, 8, 6); // 17.

		this.luckySum(3, 8, 7); // 18.

		this.luckySum(3, 8, 8); // 19.

		this.luckySum(3, 8, 9); // 20.

		this.luckySum(3, 9, 0); // 12.

		this.luckySum(3, 9, 1); // 13.

		this.luckySum(3, 9, 2); // 14.

		this.luckySum(3, 9, 3); // 15.

		this.luckySum(3, 9, 4); // 16.

		this.luckySum(3, 9, 5); // 17.

		this.luckySum(3, 9, 6); // 18.

		this.luckySum(3, 9, 7); // 19.

		this.luckySum(3, 9, 8); // 20.

		this.luckySum(3, 9, 9); // 21.

		this.luckySum(4, 0, 0); // 4.

		this.luckySum(4, 0, 1); // 5.

		this.luckySum(4, 0, 2); // 6.

		this.luckySum(4, 0, 3); // 7.

		this.luckySum(4, 0, 4); // 8.

		this.luckySum(4, 0, 5); // 9.

		this.luckySum(4, 0, 6); // 10.

		this.luckySum(4, 0, 7); // 11.

		this.luckySum(4, 0, 8); // 12.

		this.luckySum(4, 0, 9); // 13.

		this.luckySum(4, 1, 0); // 5.

		this.luckySum(4, 1, 1); // 6.

		this.luckySum(4, 1, 2); // 7.

		this.luckySum(4, 1, 3); // 8.

		this.luckySum(4, 1, 4); // 9.

		this.luckySum(4, 1, 5); // 10.

		this.luckySum(4, 1, 6); // 11.

		this.luckySum(4, 1, 7); // 12.

		this.luckySum(4, 1, 8); // 13.

		this.luckySum(4, 1, 9); // 14.

		this.luckySum(4, 2, 0); // 6.

		this.luckySum(4, 2, 1); // 7.

		this.luckySum(4, 2, 2); // 8.

		this.luckySum(4, 2, 3); // 9.

		this.luckySum(4, 2, 4); // 10.

		this.luckySum(4, 2, 5); // 11.

		this.luckySum(4, 2, 6); // 12.

		this.luckySum(4, 2, 7); // 13.

		this.luckySum(4, 2, 8); // 14.

		this.luckySum(4, 2, 9); // 15.

		this.luckySum(4, 3, 0); // 7.

		this.luckySum(4, 3, 1); // 8.

		this.luckySum(4, 3, 2); // 9.

		this.luckySum(4, 3, 3); // 10.

		this.luckySum(4, 3, 4); // 11.

		this.luckySum(4, 3, 5); // 12.

		this.luckySum(4, 3, 6); // 13.

		this.luckySum(4, 3, 7); // 14.

		this.luckySum(4, 3, 8); // 15.

		this.luckySum(4, 3, 9); // 16.

		this.luckySum(4, 4, 0); // 8.

		this.luckySum(4, 4, 1); // 9.

		this.luckySum(4, 4, 2); // 10.

		this.luckySum(4, 4, 3); // 11.

		this.luckySum(4, 4, 4); // 12.

		this.luckySum(4, 4, 5); // 13.

		this.luckySum(4, 4, 6); // 14.

		this.luckySum(4, 4, 7); // 15.

		this.luckySum(4, 4, 8); // 16.

		this.luckySum(4, 4, 9); // 17.

		this.luckySum(4, 5, 0); // 9.

		this.luckySum(4, 5, 1); // 10.

		this.luckySum(4, 5, 2); // 11.

		this.luckySum(4, 5, 3); // 12.

		this.luckySum(4, 5, 4); // 13.

		this.luckySum(4, 5, 5); // 14.

		this.luckySum(4, 5, 6); // 15.

		this.luckySum(4, 5, 7); // 16.

		this.luckySum(4, 5, 8); // 17.

		this.luckySum(4, 5, 9); // 18.

		this.luckySum(4, 6, 0); // 10.

		this.luckySum(4, 6, 1); // 11.

		this.luckySum(4, 6, 2); // 12.

		this.luckySum(4, 6, 3); // 13.

		this.luckySum(4, 6, 4); // 14.

		this.luckySum(4, 6, 5); // 15.

		this.luckySum(4, 6, 6); // 16.

		this.luckySum(4, 6, 7); // 17.

		this.luckySum(4, 6, 8); // 18.

		this.luckySum(4, 6, 9); // 19.

		this.luckySum(4, 7, 0); // 11.

		this.luckySum(4, 7, 1); // 12.

		this.luckySum(4, 7, 2); // 13.

		this.luckySum(4, 7, 3); // 14.

		this.luckySum(4, 7, 4); // 15.

		this.luckySum(4, 7, 5); // 16.

		this.luckySum(4, 7, 6); // 17.

		this.luckySum(4, 7, 7); // 18.

		this.luckySum(4, 7, 8); // 19.

		this.luckySum(4, 7, 9); // 20.

		this.luckySum(4, 8, 0); // 12.

		this.luckySum(4, 8, 1); // 13.

		this.luckySum(4, 8, 2); // 14.

		this.luckySum(4, 8, 3); // 15.

		this.luckySum(4, 8, 4); // 16.

		this.luckySum(4, 8, 5); // 17.

		this.luckySum(4, 8, 6); // 18.

		this.luckySum(4, 8, 7); // 19.

		this.luckySum(4, 8, 8); // 20.

		this.luckySum(4, 8, 9); // 21.

		this.luckySum(4, 9, 0); // 13.

		this.luckySum(4, 9, 1); // 14.

		this.luckySum(4, 9, 2); // 15.

		this.luckySum(4, 9, 3); // 16.

		this.luckySum(4, 9, 4); // 17.

		this.luckySum(4, 9, 5); // 18.

		this.luckySum(4, 9, 6); // 19.

		this.luckySum(4, 9, 7); // 20.

		this.luckySum(4, 9, 8); // 21.

		this.luckySum(4, 9, 9); // 22.

		this.luckySum(5, 0, 0); // 5.

		this.luckySum(5, 0, 1); // 6.

		this.luckySum(5, 0, 2); // 7.

		this.luckySum(5, 0, 3); // 8.

		this.luckySum(5, 0, 4); // 9.

		this.luckySum(5, 0, 5); // 10.

		this.luckySum(5, 0, 6); // 11.

		this.luckySum(5, 0, 7); // 12.

		this.luckySum(5, 0, 8); // 13.

		this.luckySum(5, 0, 9); // 14.

		this.luckySum(5, 1, 0); // 6.

		this.luckySum(5, 1, 1); // 7.

		this.luckySum(5, 1, 2); // 8.

		this.luckySum(5, 1, 3); // 9.

		this.luckySum(5, 1, 4); // 10.

		this.luckySum(5, 1, 5); // 11.

		this.luckySum(5, 1, 6); // 12.

		this.luckySum(5, 1, 7); // 13.

		this.luckySum(5, 1, 8); // 14.

		this.luckySum(5, 1, 9); // 15.

		this.luckySum(5, 2, 0); // 7.

		this.luckySum(5, 2, 1); // 8.

		this.luckySum(5, 2, 2); // 9.

		this.luckySum(5, 2, 3); // 10.

		this.luckySum(5, 2, 4); // 11.

		this.luckySum(5, 2, 5); // 12.

		this.luckySum(5, 2, 6); // 13.

		this.luckySum(5, 2, 7); // 14.

		this.luckySum(5, 2, 8); // 15.

		this.luckySum(5, 2, 9); // 16.

		this.luckySum(5, 3, 0); // 8.

		this.luckySum(5, 3, 1); // 9.

		this.luckySum(5, 3, 2); // 10.

		this.luckySum(5, 3, 3); // 11.

		this.luckySum(5, 3, 4); // 12.

		this.luckySum(5, 3, 5); // 13.

		this.luckySum(5, 3, 6); // 14.

		this.luckySum(5, 3, 7); // 15.

		this.luckySum(5, 3, 8); // 16.

		this.luckySum(5, 3, 9); // 17.

		this.luckySum(5, 4, 0); // 9.

		this.luckySum(5, 4, 1); // 10.

		this.luckySum(5, 4, 2); // 11.

		this.luckySum(5, 4, 3); // 12.

		this.luckySum(5, 4, 4); // 13.

		this.luckySum(5, 4, 5); // 14.

		this.luckySum(5, 4, 6); // 15.

		this.luckySum(5, 4, 7); // 16.

		this.luckySum(5, 4, 8); // 17.

		this.luckySum(5, 4, 9); // 18.

		this.luckySum(5, 5, 0); // 10.

		this.luckySum(5, 5, 1); // 11.

		this.luckySum(5, 5, 2); // 12.

		this.luckySum(5, 5, 3); // 13.

		this.luckySum(5, 5, 4); // 14.

		this.luckySum(5, 5, 5); // 15.

		this.luckySum(5, 5, 6); // 16.

		this.luckySum(5, 5, 7); // 17.

		this.luckySum(5, 5, 8); // 18.

		this.luckySum(5, 5, 9); // 19.

		this.luckySum(5, 6, 0); // 11.

		this.luckySum(5, 6, 1); // 12.

		this.luckySum(5, 6, 2); // 13.

		this.luckySum(5, 6, 3); // 14.

		this.luckySum(5, 6, 4); // 15.

		this.luckySum(5, 6, 5); // 16.

		this.luckySum(5, 6, 6); // 17.

		this.luckySum(5, 6, 7); // 18.

		this.luckySum(5, 6, 8); // 19.

		this.luckySum(5, 6, 9); // 20.

		this.luckySum(5, 7, 0); // 12.

		this.luckySum(5, 7, 1); // 13.

		this.luckySum(5, 7, 2); // 14.

		this.luckySum(5, 7, 3); // 15.

		this.luckySum(5, 7, 4); // 16.

		this.luckySum(5, 7, 5); // 17.

		this.luckySum(5, 7, 6); // 18.

		this.luckySum(5, 7, 7); // 19.

		this.luckySum(5, 7, 8); // 20.

		this.luckySum(5, 7, 9); // 21.

		this.luckySum(5, 8, 0); // 13.

		this.luckySum(5, 8, 1); // 14.

		this.luckySum(5, 8, 2); // 15.

		this.luckySum(5, 8, 3); // 16.

		this.luckySum(5, 8, 4); // 17.

		this.luckySum(5, 8, 5); // 18.

		this.luckySum(5, 8, 6); // 19.

		this.luckySum(5, 8, 7); // 20.

		this.luckySum(5, 8, 8); // 21.

		this.luckySum(5, 8, 9); // 22.

		this.luckySum(5, 9, 0); // 14.

		this.luckySum(5, 9, 1); // 15.

		this.luckySum(5, 9, 2); // 16.

		this.luckySum(5, 9, 3); // 17.

		this.luckySum(5, 9, 4); // 18.

		this.luckySum(5, 9, 5); // 19.

		this.luckySum(5, 9, 6); // 20.

		this.luckySum(5, 9, 7); // 21.

		this.luckySum(5, 9, 8); // 22.

		this.luckySum(5, 9, 9); // 23.

		this.luckySum(6, 0, 0); // 6.

		this.luckySum(6, 0, 1); // 7.

		this.luckySum(6, 0, 2); // 8.

		this.luckySum(6, 0, 3); // 9.

		this.luckySum(6, 0, 4); // 10.

		this.luckySum(6, 0, 5); // 11.

		this.luckySum(6, 0, 6); // 12.

		this.luckySum(6, 0, 7); // 13.

		this.luckySum(6, 0, 8); // 14.

		this.luckySum(6, 0, 9); // 15.

		this.luckySum(6, 1, 0); // 7.

		this.luckySum(6, 1, 1); // 8.

		this.luckySum(6, 1, 2); // 9.

		this.luckySum(6, 1, 3); // 10.

		this.luckySum(6, 1, 4); // 11.

		this.luckySum(6, 1, 5); // 12.

		this.luckySum(6, 1, 6); // 13.

		this.luckySum(6, 1, 7); // 14.

		this.luckySum(6, 1, 8); // 15.

		this.luckySum(6, 1, 9); // 16.

		this.luckySum(6, 2, 0); // 8.

		this.luckySum(6, 2, 1); // 9.

		this.luckySum(6, 2, 2); // 10.

		this.luckySum(6, 2, 3); // 11.

		this.luckySum(6, 2, 4); // 12.

		this.luckySum(6, 2, 5); // 13.

		this.luckySum(6, 2, 6); // 14.

		this.luckySum(6, 2, 7); // 15.

		this.luckySum(6, 2, 8); // 16.

		this.luckySum(6, 2, 9); // 17.

		this.luckySum(6, 3, 0); // 9.

		this.luckySum(6, 3, 1); // 10.

		this.luckySum(6, 3, 2); // 11.

		this.luckySum(6, 3, 3); // 12.

		this.luckySum(6, 3, 4); // 13.

		this.luckySum(6, 3, 5); // 14.

		this.luckySum(6, 3, 6); // 15.

		this.luckySum(6, 3, 7); // 16.

		this.luckySum(6, 3, 8); // 17.

		this.luckySum(6, 3, 9); // 18.

		this.luckySum(6, 4, 0); // 10.

		this.luckySum(6, 4, 1); // 11.

		this.luckySum(6, 4, 2); // 12.

		this.luckySum(6, 4, 3); // 13.

		this.luckySum(6, 4, 4); // 14.

		this.luckySum(6, 4, 5); // 15.

		this.luckySum(6, 4, 6); // 16.

		this.luckySum(6, 4, 7); // 17.

		this.luckySum(6, 4, 8); // 18.

		this.luckySum(6, 4, 9); // 19.

		this.luckySum(6, 5, 0); // 11.

		this.luckySum(6, 5, 1); // 12.

		this.luckySum(6, 5, 2); // 13.

		this.luckySum(6, 5, 3); // 14.

		this.luckySum(6, 5, 4); // 15.

		this.luckySum(6, 5, 5); // 16.

		this.luckySum(6, 5, 6); // 17.

		this.luckySum(6, 5, 7); // 18.

		this.luckySum(6, 5, 8); // 19.

		this.luckySum(6, 5, 9); // 20.

		this.luckySum(6, 6, 0); // 12.

		this.luckySum(6, 6, 1); // 13.

		this.luckySum(6, 6, 2); // 14.

		this.luckySum(6, 6, 3); // 15.

		this.luckySum(6, 6, 4); // 16.

		this.luckySum(6, 6, 5); // 17.

		this.luckySum(6, 6, 6); // 18.

		this.luckySum(6, 6, 7); // 19.

		this.luckySum(6, 6, 8); // 20.

		this.luckySum(6, 6, 9); // 21.

		this.luckySum(6, 7, 0); // 13.

		this.luckySum(6, 7, 1); // 14.

		this.luckySum(6, 7, 2); // 15.

		this.luckySum(6, 7, 3); // 16.

		this.luckySum(6, 7, 4); // 17.

		this.luckySum(6, 7, 5); // 18.

		this.luckySum(6, 7, 6); // 19.

		this.luckySum(6, 7, 7); // 20.

		this.luckySum(6, 7, 8); // 21.

		this.luckySum(6, 7, 9); // 22.

		this.luckySum(6, 8, 0); // 14.

		this.luckySum(6, 8, 1); // 15.

		this.luckySum(6, 8, 2); // 16.

		this.luckySum(6, 8, 3); // 17.

		this.luckySum(6, 8, 4); // 18.

		this.luckySum(6, 8, 5); // 19.

		this.luckySum(6, 8, 6); // 20.

		this.luckySum(6, 8, 7); // 21.

		this.luckySum(6, 8, 8); // 22.

		this.luckySum(6, 8, 9); // 23.

		this.luckySum(6, 9, 0); // 15.

		this.luckySum(6, 9, 1); // 16.

		this.luckySum(6, 9, 2); // 17.

		this.luckySum(6, 9, 3); // 18.

		this.luckySum(6, 9, 4); // 19.

		this.luckySum(6, 9, 5); // 20.

		this.luckySum(6, 9, 6); // 21.

		this.luckySum(6, 9, 7); // 22.

		this.luckySum(6, 9, 8); // 23.

		this.luckySum(6, 9, 9); // 24.

		this.luckySum(7, 0, 0); // 7.

		this.luckySum(7, 0, 1); // 8.

		this.luckySum(7, 0, 2); // 9.

		this.luckySum(7, 0, 3); // 10.

		this.luckySum(7, 0, 4); // 11.

		this.luckySum(7, 0, 5); // 12.

		this.luckySum(7, 0, 6); // 13.

		this.luckySum(7, 0, 7); // 14.

		this.luckySum(7, 0, 8); // 15.

		this.luckySum(7, 0, 9); // 16.

		this.luckySum(7, 1, 0); // 8.

		this.luckySum(7, 1, 1); // 9.

		this.luckySum(7, 1, 2); // 10.

		this.luckySum(7, 1, 3); // 11.

		this.luckySum(7, 1, 4); // 12.

		this.luckySum(7, 1, 5); // 13.

		this.luckySum(7, 1, 6); // 14.

		this.luckySum(7, 1, 7); // 15.

		this.luckySum(7, 1, 8); // 16.

		this.luckySum(7, 1, 9); // 17.

		this.luckySum(7, 2, 0); // 9.

		this.luckySum(7, 2, 1); // 10.

		this.luckySum(7, 2, 2); // 11.

		this.luckySum(7, 2, 3); // 12.

		this.luckySum(7, 2, 4); // 13.

		this.luckySum(7, 2, 5); // 14.

		this.luckySum(7, 2, 6); // 15.

		this.luckySum(7, 2, 7); // 16.

		this.luckySum(7, 2, 8); // 17.

		this.luckySum(7, 2, 9); // 18.

		this.luckySum(7, 3, 0); // 10.

		this.luckySum(7, 3, 1); // 11.

		this.luckySum(7, 3, 2); // 12.

		this.luckySum(7, 3, 3); // 13.

		this.luckySum(7, 3, 4); // 14.

		this.luckySum(7, 3, 5); // 15.

		this.luckySum(7, 3, 6); // 16.

		this.luckySum(7, 3, 7); // 17.

		this.luckySum(7, 3, 8); // 18.

		this.luckySum(7, 3, 9); // 19.

		this.luckySum(7, 4, 0); // 11.

		this.luckySum(7, 4, 1); // 12.

		this.luckySum(7, 4, 2); // 13.

		this.luckySum(7, 4, 3); // 14.

		this.luckySum(7, 4, 4); // 15.

		this.luckySum(7, 4, 5); // 16.

		this.luckySum(7, 4, 6); // 17.

		this.luckySum(7, 4, 7); // 18.

		this.luckySum(7, 4, 8); // 19.

		this.luckySum(7, 4, 9); // 20.

		this.luckySum(7, 5, 0); // 12.

		this.luckySum(7, 5, 1); // 13.

		this.luckySum(7, 5, 2); // 14.

		this.luckySum(7, 5, 3); // 15.

		this.luckySum(7, 5, 4); // 16.

		this.luckySum(7, 5, 5); // 17.

		this.luckySum(7, 5, 6); // 18.

		this.luckySum(7, 5, 7); // 19.

		this.luckySum(7, 5, 8); // 20.

		this.luckySum(7, 5, 9); // 21.

		this.luckySum(7, 6, 0); // 13.

		this.luckySum(7, 6, 1); // 14.

		this.luckySum(7, 6, 2); // 15.

		this.luckySum(7, 6, 3); // 16.

		this.luckySum(7, 6, 4); // 17.

		this.luckySum(7, 6, 5); // 18.

		this.luckySum(7, 6, 6); // 19.

		this.luckySum(7, 6, 7); // 20.

		this.luckySum(7, 6, 8); // 21.

		this.luckySum(7, 6, 9); // 22.

		this.luckySum(7, 7, 0); // 14.

		this.luckySum(7, 7, 1); // 15.

		this.luckySum(7, 7, 2); // 16.

		this.luckySum(7, 7, 3); // 17.

		this.luckySum(7, 7, 4); // 18.

		this.luckySum(7, 7, 5); // 19.

		this.luckySum(7, 7, 6); // 20.

		this.luckySum(7, 7, 7); // 21.

		this.luckySum(7, 7, 8); // 22.

		this.luckySum(7, 7, 9); // 23.

		this.luckySum(7, 8, 0); // 15.

		this.luckySum(7, 8, 1); // 16.

		this.luckySum(7, 8, 2); // 17.

		this.luckySum(7, 8, 3); // 18.

		this.luckySum(7, 8, 4); // 19.

		this.luckySum(7, 8, 5); // 20.

		this.luckySum(7, 8, 6); // 21.

		this.luckySum(7, 8, 7); // 22.

		this.luckySum(7, 8, 8); // 23.

		this.luckySum(7, 8, 9); // 24.

		this.luckySum(7, 9, 0); // 16.

		this.luckySum(7, 9, 1); // 17.

		this.luckySum(7, 9, 2); // 18.

		this.luckySum(7, 9, 3); // 19.

		this.luckySum(7, 9, 4); // 20.

		this.luckySum(7, 9, 5); // 21.

		this.luckySum(7, 9, 6); // 22.

		this.luckySum(7, 9, 7); // 23.

		this.luckySum(7, 9, 8); // 24.

		this.luckySum(7, 9, 9); // 25.

		this.luckySum(8, 0, 0); // 8.

		this.luckySum(8, 0, 1); // 9.

		this.luckySum(8, 0, 2); // 10.

		this.luckySum(8, 0, 3); // 11.

		this.luckySum(8, 0, 4); // 12.

		this.luckySum(8, 0, 5); // 13.

		this.luckySum(8, 0, 6); // 14.

		this.luckySum(8, 0, 7); // 15.

		this.luckySum(8, 0, 8); // 16.

		this.luckySum(8, 0, 9); // 17.

		this.luckySum(8, 1, 0); // 9.

		this.luckySum(8, 1, 1); // 10.

		this.luckySum(8, 1, 2); // 11.

		this.luckySum(8, 1, 3); // 12.

		this.luckySum(8, 1, 4); // 13.

		this.luckySum(8, 1, 5); // 14.

		this.luckySum(8, 1, 6); // 15.

		this.luckySum(8, 1, 7); // 16.

		this.luckySum(8, 1, 8); // 17.

		this.luckySum(8, 1, 9); // 18.

		this.luckySum(8, 2, 0); // 10.

		this.luckySum(8, 2, 1); // 11.

		this.luckySum(8, 2, 2); // 12.

		this.luckySum(8, 2, 3); // 13.

		this.luckySum(8, 2, 4); // 14.

		this.luckySum(8, 2, 5); // 15.

		this.luckySum(8, 2, 6); // 16.

		this.luckySum(8, 2, 7); // 17.

		this.luckySum(8, 2, 8); // 18.

		this.luckySum(8, 2, 9); // 19.

		this.luckySum(8, 3, 0); // 11.

		this.luckySum(8, 3, 1); // 12.

		this.luckySum(8, 3, 2); // 13.

		this.luckySum(8, 3, 3); // 14.

		this.luckySum(8, 3, 4); // 15.

		this.luckySum(8, 3, 5); // 16.

		this.luckySum(8, 3, 6); // 17.

		this.luckySum(8, 3, 7); // 18.

		this.luckySum(8, 3, 8); // 19.

		this.luckySum(8, 3, 9); // 20.

		this.luckySum(8, 4, 0); // 12.

		this.luckySum(8, 4, 1); // 13.

		this.luckySum(8, 4, 2); // 14.

		this.luckySum(8, 4, 3); // 15.

		this.luckySum(8, 4, 4); // 16.

		this.luckySum(8, 4, 5); // 17.

		this.luckySum(8, 4, 6); // 18.

		this.luckySum(8, 4, 7); // 19.

		this.luckySum(8, 4, 8); // 20.

		this.luckySum(8, 4, 9); // 21.

		this.luckySum(8, 5, 0); // 13.

		this.luckySum(8, 5, 1); // 14.

		this.luckySum(8, 5, 2); // 15.

		this.luckySum(8, 5, 3); // 16.

		this.luckySum(8, 5, 4); // 17.

		this.luckySum(8, 5, 5); // 18.

		this.luckySum(8, 5, 6); // 19.

		this.luckySum(8, 5, 7); // 20.

		this.luckySum(8, 5, 8); // 21.

		this.luckySum(8, 5, 9); // 22.

		this.luckySum(8, 6, 0); // 14.

		this.luckySum(8, 6, 1); // 15.

		this.luckySum(8, 6, 2); // 16.

		this.luckySum(8, 6, 3); // 17.

		this.luckySum(8, 6, 4); // 18.

		this.luckySum(8, 6, 5); // 19.

		this.luckySum(8, 6, 6); // 20.

		this.luckySum(8, 6, 7); // 21.

		this.luckySum(8, 6, 8); // 22.

		this.luckySum(8, 6, 9); // 23.

		this.luckySum(8, 7, 0); // 15.

		this.luckySum(8, 7, 1); // 16.

		this.luckySum(8, 7, 2); // 17.

		this.luckySum(8, 7, 3); // 18.

		this.luckySum(8, 7, 4); // 19.

		this.luckySum(8, 7, 5); // 20.

		this.luckySum(8, 7, 6); // 21.

		this.luckySum(8, 7, 7); // 22.

		this.luckySum(8, 7, 8); // 23.

		this.luckySum(8, 7, 9); // 24.

		this.luckySum(8, 8, 0); // 16.

		this.luckySum(8, 8, 1); // 17.

		this.luckySum(8, 8, 2); // 18.

		this.luckySum(8, 8, 3); // 19.

		this.luckySum(8, 8, 4); // 20.

		this.luckySum(8, 8, 5); // 21.

		this.luckySum(8, 8, 6); // 22.

		this.luckySum(8, 8, 7); // 23.

		this.luckySum(8, 8, 8); // 24.

		this.luckySum(8, 8, 9); // 25.

		this.luckySum(8, 9, 0); // 17.

		this.luckySum(8, 9, 1); // 18.

		this.luckySum(8, 9, 2); // 19.

		this.luckySum(8, 9, 3); // 20.

		this.luckySum(8, 9, 4); // 21.

		this.luckySum(8, 9, 5); // 22.

		this.luckySum(8, 9, 6); // 23.

		this.luckySum(8, 9, 7); // 24.

		this.luckySum(8, 9, 8); // 25.

		this.luckySum(8, 9, 9); // 26.

		this.luckySum(9, 0, 0); // 9.

		this.luckySum(9, 0, 1); // 10.

		this.luckySum(9, 0, 2); // 11.

		this.luckySum(9, 0, 3); // 12.

		this.luckySum(9, 0, 4); // 13.

		this.luckySum(9, 0, 5); // 14.

		this.luckySum(9, 0, 6); // 15.

		this.luckySum(9, 0, 7); // 16.

		this.luckySum(9, 0, 8); // 17.

		this.luckySum(9, 0, 9); // 18.

		this.luckySum(9, 1, 0); // 10.

		this.luckySum(9, 1, 1); // 11.

		this.luckySum(9, 1, 2); // 12.

		this.luckySum(9, 1, 3); // 13.

		this.luckySum(9, 1, 4); // 14.

		this.luckySum(9, 1, 5); // 15.

		this.luckySum(9, 1, 6); // 16.

		this.luckySum(9, 1, 7); // 17.

		this.luckySum(9, 1, 8); // 18.

		this.luckySum(9, 1, 9); // 19.

		this.luckySum(9, 2, 0); // 11.

		this.luckySum(9, 2, 1); // 12.

		this.luckySum(9, 2, 2); // 13.

		this.luckySum(9, 2, 3); // 14.

		this.luckySum(9, 2, 4); // 15.

		this.luckySum(9, 2, 5); // 16.

		this.luckySum(9, 2, 6); // 17.

		this.luckySum(9, 2, 7); // 18.

		this.luckySum(9, 2, 8); // 19.

		this.luckySum(9, 2, 9); // 20.

		this.luckySum(9, 3, 0); // 12.

		this.luckySum(9, 3, 1); // 13.

		this.luckySum(9, 3, 2); // 14.

		this.luckySum(9, 3, 3); // 15.

		this.luckySum(9, 3, 4); // 16.

		this.luckySum(9, 3, 5); // 17.

		this.luckySum(9, 3, 6); // 18.

		this.luckySum(9, 3, 7); // 19.

		this.luckySum(9, 3, 8); // 20.

		this.luckySum(9, 3, 9); // 21.

		this.luckySum(9, 4, 0); // 13.

		this.luckySum(9, 4, 1); // 14.

		this.luckySum(9, 4, 2); // 15.

		this.luckySum(9, 4, 3); // 16.

		this.luckySum(9, 4, 4); // 17.

		this.luckySum(9, 4, 5); // 18.

		this.luckySum(9, 4, 6); // 19.

		this.luckySum(9, 4, 7); // 20.

		this.luckySum(9, 4, 8); // 21.

		this.luckySum(9, 4, 9); // 22.

		this.luckySum(9, 5, 0); // 14.

		this.luckySum(9, 5, 1); // 15.

		this.luckySum(9, 5, 2); // 16.

		this.luckySum(9, 5, 3); // 17.

		this.luckySum(9, 5, 4); // 18.

		this.luckySum(9, 5, 5); // 19.

		this.luckySum(9, 5, 6); // 20.

		this.luckySum(9, 5, 7); // 21.

		this.luckySum(9, 5, 8); // 22.

		this.luckySum(9, 5, 9); // 23.

		this.luckySum(9, 6, 0); // 15.

		this.luckySum(9, 6, 1); // 16.

		this.luckySum(9, 6, 2); // 17.

		this.luckySum(9, 6, 3); // 18.

		this.luckySum(9, 6, 4); // 19.

		this.luckySum(9, 6, 5); // 20.

		this.luckySum(9, 6, 6); // 21.

		this.luckySum(9, 6, 7); // 22.

		this.luckySum(9, 6, 8); // 23.

		this.luckySum(9, 6, 9); // 24.

		this.luckySum(9, 7, 0); // 16.

		this.luckySum(9, 7, 1); // 17.

		this.luckySum(9, 7, 2); // 18.

		this.luckySum(9, 7, 3); // 19.

		this.luckySum(9, 7, 4); // 20.

		this.luckySum(9, 7, 5); // 21.

		this.luckySum(9, 7, 6); // 22.

		this.luckySum(9, 7, 7); // 23.

		this.luckySum(9, 7, 8); // 24.

		this.luckySum(9, 7, 9); // 25.

		this.luckySum(9, 8, 0); // 17.

		this.luckySum(9, 8, 1); // 18.

		this.luckySum(9, 8, 2); // 19.

		this.luckySum(9, 8, 3); // 20.

		this.luckySum(9, 8, 4); // 21.

		this.luckySum(9, 8, 5); // 22.

		this.luckySum(9, 8, 6); // 23.

		this.luckySum(9, 8, 7); // 24.

		this.luckySum(9, 8, 8); // 25.

		this.luckySum(9, 8, 9); // 26.

		this.luckySum(9, 9, 0); // 18.

		this.luckySum(9, 9, 1); // 19.

		this.luckySum(9, 9, 2); // 20.

		this.luckySum(9, 9, 3); // 21.

		this.luckySum(9, 9, 4); // 22.

		this.luckySum(9, 9, 5); // 23.

		this.luckySum(9, 9, 6); // 24.

		this.luckySum(9, 9, 7); // 25.

		this.luckySum(9, 9, 8); // 26.

		this.luckySum(9, 9, 9); // 27.



		// factorialWithFor(n) Test Cases



		this.factorialFor(1); // 1! = 1.

		this.factorialFor(2); // 2! = 2.

		this.factorialFor(3); // 3! = 6.

		this.factorialFor(4); // 4! = 24.

		this.factorialFor(5); // 5! = 120.

		this.factorialFor(6); // 6! = 720.

		this.factorialFor(7); // 7! = 5040.

		this.factorialFor(8); // 8! = 40320.

		this.factorialFor(9); // 9! = 362880.

		this.factorialFor(10); // 10! = 3628800.



		// factorialWithWhile(n) Test Cases



		this.factorialWhile(1); // 1! = 1.

		this.factorialWhile(2); // 2! = 2.

		this.factorialWhile(3); // 3! = 6.

		this.factorialWhile(4); // 4! = 24.

		this.factorialWhile(5); // 5! = 120.

		this.factorialWhile(6); // 6! = 720.

		this.factorialWhile(7); // 7! = 5040.

		this.factorialWhile(8); // 8! = 40320.

		this.factorialWhile(9); // 9! = 362880.

		this.factorialWhile(10); // 10! = 3628800.



		// isPrime(n) Test Cases



		this.isPrime(0); // NOT PRIME.

		this.isPrime(1); // NOT PRIME.

		this.isPrime(2); // PRIME.

		this.isPrime(3); // PRIME.

		this.isPrime(4); // NOT PRIME.

		this.isPrime(5); // PRIME.

		this.isPrime(6); // NOT PRIME.

		this.isPrime(7); // PRIME.

		this.isPrime(8); // NOT PRIME.

		this.isPrime(9); // NOT PRIME.

		this.isPrime(10); // NOT PRIME.

		this.isPrime(11); // PRIME.

		this.isPrime(12); // NOT PRIME.

		this.isPrime(13); // PRIME.

		this.isPrime(14); // NOT PRIME.

		this.isPrime(15); // NOT PRIME.

		this.isPrime(16); // NOT PRIME.

		this.isPrime(17); // PRIME.

		this.isPrime(18); // NOT PRIME.

		this.isPrime(19); // PRIME.

		this.isPrime(20); // NOT PRIME.

		this.isPrime(21); // NOT PRIME.

		this.isPrime(22); // NOT PRIME.

		this.isPrime(23); // PRIME.

		this.isPrime(24); // NOT PRIME.

		this.isPrime(25); // NOT PRIME.

		this.isPrime(26); // NOT PRIME.

		this.isPrime(27); // NOT PRIME.

		this.isPrime(28); // NOT PRIME.

		this.isPrime(29); // PRIME.

		this.isPrime(30); // NOT PRIME.

		this.isPrime(31); // PRIME.

		this.isPrime(32); // NOT PRIME.

		this.isPrime(33); // NOT PRIME.

		this.isPrime(34); // NOT PRIME.

		this.isPrime(35); // NOT PRIME.

		this.isPrime(36); // NOT PRIME.

		this.isPrime(37); // PRIME.

		this.isPrime(38); // NOT PRIME.

		this.isPrime(39); // NOT PRIME.

		this.isPrime(40); // NOT PRIME.

		this.isPrime(41); // PRIME.

		this.isPrime(42); // NOT PRIME.

		this.isPrime(43); // PRIME.

		this.isPrime(44); // NOT PRIME.

		this.isPrime(45); // NOT PRIME.

		this.isPrime(46); // NOT PRIME.

		this.isPrime(47); // PRIME.

		this.isPrime(48); // NOT PRIME.

		this.isPrime(49); // NOT PRIME.

		this.isPrime(50); // NOT PRIME.

		this.isPrime(51); // NOT PRIME.

		this.isPrime(52); // NOT PRIME.

		this.isPrime(53); // PRIME.

		this.isPrime(54); // NOT PRIME.

		this.isPrime(55); // NOT PRIME.

		this.isPrime(56); // NOT PRIME.

		this.isPrime(57); // NOT PRIME.

		this.isPrime(58); // NOT PRIME.

		this.isPrime(59); // PRIME.

		this.isPrime(60); // NOT PRIME.

		this.isPrime(61); // PRIME.

		this.isPrime(62); // NOT PRIME.

		this.isPrime(63); // NOT PRIME.

		this.isPrime(64); // NOT PRIME.

		this.isPrime(65); // NOT PRIME.

		this.isPrime(66); // NOT PRIME.

		this.isPrime(67); // PRIME.

		this.isPrime(68); // NOT PRIME.

		this.isPrime(69); // NOT PRIME.

		this.isPrime(70); // NOT PRIME.

		this.isPrime(71); // PRIME.

		this.isPrime(72); // NOT PRIME.

		this.isPrime(73); // PRIME.

		this.isPrime(74); // NOT PRIME.

		this.isPrime(75); // NOT PRIME.

		this.isPrime(76); // NOT PRIME.

		this.isPrime(77); // NOT PRIME.

		this.isPrime(78); // NOT PRIME.

		this.isPrime(79); // PRIME.

		this.isPrime(80); // NOT PRIME.

		this.isPrime(81); // NOT PRIME.

		this.isPrime(82); // NOT PRIME.

		this.isPrime(83); // PRIME.

		this.isPrime(84); // NOT PRIME.

		this.isPrime(85); // NOT PRIME.

		this.isPrime(86); // NOT PRIME.

		this.isPrime(87); // NOT PRIME.

		this.isPrime(88); // NOT PRIME.

		this.isPrime(89); // PRIME.

		this.isPrime(90); // NOT PRIME.

		this.isPrime(91); // NOT PRIME.

		this.isPrime(92); // NOT PRIME.

		this.isPrime(93); // NOT PRIME.

		this.isPrime(94); // NOT PRIME.

		this.isPrime(95); // NOT PRIME.

		this.isPrime(96); // NOT PRIME.

		this.isPrime(97); // PRIME.

		this.isPrime(98); // NOT PRIME.

		this.isPrime(99); // NOT PRIME.

		this.isPrime(100); // NOT PRIME.

		this.isPrime(101); // PRIME.

		this.isPrime(102); // NOT PRIME.

		this.isPrime(103); // PRIME.

		this.isPrime(104); // NOT PRIME.

		this.isPrime(105); // NOT PRIME.

		this.isPrime(106); // NOT PRIME.

		this.isPrime(107); // PRIME.

		this.isPrime(108); // NOT PRIME.

		this.isPrime(109); // PRIME.

		this.isPrime(110); // NOT PRIME.

		this.isPrime(111); // NOT PRIME.

		this.isPrime(112); // NOT PRIME.

		this.isPrime(113); // PRIME.

		this.isPrime(114); // NOT PRIME.

		this.isPrime(115); // NOT PRIME.

		this.isPrime(116); // NOT PRIME.

		this.isPrime(117); // NOT PRIME.

		this.isPrime(118); // NOT PRIME.

		this.isPrime(119); // NOT PRIME.

		this.isPrime(120); // NOT PRIME.

		this.isPrime(121); // NOT PRIME.

		this.isPrime(122); // NOT PRIME.

		this.isPrime(123); // NOT PRIME.

		this.isPrime(124); // NOT PRIME.

		this.isPrime(125); // NOT PRIME.

		this.isPrime(126); // NOT PRIME.

		this.isPrime(127); // PRIME.

		this.isPrime(128); // NOT PRIME.

		this.isPrime(129); // NOT PRIME.

		this.isPrime(130); // NOT PRIME.

		this.isPrime(131); // PRIME.

		this.isPrime(132); // NOT PRIME.

		this.isPrime(133); // NOT PRIME.

		this.isPrime(134); // NOT PRIME.

		this.isPrime(135); // NOT PRIME.

		this.isPrime(136); // NOT PRIME.

		this.isPrime(137); // PRIME.

		this.isPrime(138); // NOT PRIME.

		this.isPrime(139); // PRIME.

		this.isPrime(140); // NOT PRIME.

		this.isPrime(141); // NOT PRIME.

		this.isPrime(142); // NOT PRIME.

		this.isPrime(143); // NOT PRIME.

		this.isPrime(144); // NOT PRIME.

		this.isPrime(145); // NOT PRIME.

		this.isPrime(146); // NOT PRIME.

		this.isPrime(147); // NOT PRIME.

		this.isPrime(148); // NOT PRIME.

		this.isPrime(149); // PRIME.

		this.isPrime(150); // NOT PRIME.

		this.isPrime(151); // PRIME.

		this.isPrime(152); // NOT PRIME.

		this.isPrime(153); // NOT PRIME.

		this.isPrime(154); // NOT PRIME.

		this.isPrime(155); // NOT PRIME.

		this.isPrime(156); // NOT PRIME.

		this.isPrime(157); // PRIME.

		this.isPrime(158); // NOT PRIME.

		this.isPrime(159); // NOT PRIME.

		this.isPrime(160); // NOT PRIME.

		this.isPrime(161); // NOT PRIME.

		this.isPrime(162); // NOT PRIME.

		this.isPrime(163); // PRIME.

		this.isPrime(164); // NOT PRIME.

		this.isPrime(165); // NOT PRIME.

		this.isPrime(166); // NOT PRIME.

		this.isPrime(167); // PRIME.

		this.isPrime(168); // NOT PRIME.

		this.isPrime(169); // NOT PRIME.

		this.isPrime(170); // NOT PRIME.

		this.isPrime(171); // NOT PRIME.

		this.isPrime(172); // NOT PRIME.

		this.isPrime(173); // PRIME.

		this.isPrime(174); // NOT PRIME.

		this.isPrime(175); // NOT PRIME.

		this.isPrime(176); // NOT PRIME.

		this.isPrime(177); // NOT PRIME.

		this.isPrime(178); // NOT PRIME.

		this.isPrime(179); // PRIME.

		this.isPrime(180); // NOT PRIME.

		this.isPrime(181); // PRIME.

		this.isPrime(182); // NOT PRIME.

		this.isPrime(183); // NOT PRIME.

		this.isPrime(184); // NOT PRIME.

		this.isPrime(185); // NOT PRIME.

		this.isPrime(186); // NOT PRIME.

		this.isPrime(187); // NOT PRIME.

		this.isPrime(188); // NOT PRIME.

		this.isPrime(189); // NOT PRIME.

		this.isPrime(190); // NOT PRIME.

		this.isPrime(191); // PRIME.

		this.isPrime(192); // NOT PRIME.

		this.isPrime(193); // PRIME.

		this.isPrime(194); // NOT PRIME.

		this.isPrime(195); // NOT PRIME.

		this.isPrime(196); // NOT PRIME.

		this.isPrime(197); // PRIME.

		this.isPrime(198); // NOT PRIME.

		this.isPrime(199); // PRIME.

		this.isPrime(200); // NOT PRIME.

		this.isPrime(201); // NOT PRIME.

		this.isPrime(202); // NOT PRIME.

		this.isPrime(203); // NOT PRIME.

		this.isPrime(204); // NOT PRIME.

		this.isPrime(205); // NOT PRIME.

		this.isPrime(206); // NOT PRIME.

		this.isPrime(207); // NOT PRIME.

		this.isPrime(208); // NOT PRIME.

		this.isPrime(209); // NOT PRIME.

		this.isPrime(210); // NOT PRIME.

		this.isPrime(211); // PRIME.

		this.isPrime(212); // NOT PRIME.

		this.isPrime(213); // NOT PRIME.

		this.isPrime(214); // NOT PRIME.

		this.isPrime(215); // NOT PRIME.

		this.isPrime(216); // NOT PRIME.

		this.isPrime(217); // NOT PRIME.

		this.isPrime(218); // NOT PRIME.

		this.isPrime(219); // NOT PRIME.

		this.isPrime(220); // NOT PRIME.

		this.isPrime(221); // NOT PRIME.

		this.isPrime(222); // NOT PRIME.

		this.isPrime(223); // PRIME.

		this.isPrime(224); // NOT PRIME.

		this.isPrime(225); // NOT PRIME.

		this.isPrime(226); // NOT PRIME.

		this.isPrime(227); // PRIME.

		this.isPrime(228); // NOT PRIME.

		this.isPrime(229); // PRIME.

		this.isPrime(230); // NOT PRIME.

		this.isPrime(231); // NOT PRIME.

		this.isPrime(232); // NOT PRIME.

		this.isPrime(233); // PRIME.

		this.isPrime(234); // NOT PRIME.

		this.isPrime(235); // NOT PRIME.

		this.isPrime(236); // NOT PRIME.

		this.isPrime(237); // NOT PRIME.

		this.isPrime(238); // NOT PRIME.

		this.isPrime(239); // PRIME.

		this.isPrime(240); // NOT PRIME.

		this.isPrime(241); // PRIME.

		this.isPrime(242); // NOT PRIME.

		this.isPrime(243); // NOT PRIME.

		this.isPrime(244); // NOT PRIME.

		this.isPrime(245); // NOT PRIME.

		this.isPrime(246); // NOT PRIME.

		this.isPrime(247); // NOT PRIME.

		this.isPrime(248); // NOT PRIME.

		this.isPrime(249); // NOT PRIME.

		this.isPrime(250); // NOT PRIME.

		this.isPrime(251); // PRIME.

		this.isPrime(252); // NOT PRIME.

		this.isPrime(253); // NOT PRIME.

		this.isPrime(254); // NOT PRIME.

		this.isPrime(255); // NOT PRIME.

		this.isPrime(256); // NOT PRIME.

		this.isPrime(257); // PRIME.

		this.isPrime(258); // NOT PRIME.

		this.isPrime(259); // NOT PRIME.

		this.isPrime(260); // NOT PRIME.

		this.isPrime(261); // NOT PRIME.

		this.isPrime(262); // NOT PRIME.

		this.isPrime(263); // PRIME.

		this.isPrime(264); // NOT PRIME.

		this.isPrime(265); // NOT PRIME.

		this.isPrime(266); // NOT PRIME.

		this.isPrime(267); // NOT PRIME.

		this.isPrime(268); // NOT PRIME.

		this.isPrime(269); // PRIME.

		this.isPrime(270); // NOT PRIME.

		this.isPrime(271); // PRIME.

		this.isPrime(272); // NOT PRIME.

		this.isPrime(273); // NOT PRIME.

		this.isPrime(274); // NOT PRIME.

		this.isPrime(275); // NOT PRIME.

		this.isPrime(276); // NOT PRIME.

		this.isPrime(277); // PRIME.

		this.isPrime(278); // NOT PRIME.

		this.isPrime(279); // NOT PRIME.

		this.isPrime(280); // NOT PRIME.

		this.isPrime(281); // PRIME.

		this.isPrime(282); // NOT PRIME.

		this.isPrime(283); // PRIME.

		this.isPrime(284); // NOT PRIME.

		this.isPrime(285); // NOT PRIME.

		this.isPrime(286); // NOT PRIME.

		this.isPrime(287); // NOT PRIME.

		this.isPrime(288); // NOT PRIME.

		this.isPrime(289); // NOT PRIME.

		this.isPrime(290); // NOT PRIME.

		this.isPrime(291); // NOT PRIME.

		this.isPrime(292); // NOT PRIME.

		this.isPrime(293); // PRIME.

		this.isPrime(294); // NOT PRIME.

		this.isPrime(295); // NOT PRIME.

		this.isPrime(296); // NOT PRIME.

		this.isPrime(297); // NOT PRIME.

		this.isPrime(298); // NOT PRIME.

		this.isPrime(299); // NOT PRIME.

		this.isPrime(300); // NOT PRIME.

		this.isPrime(301); // NOT PRIME.

		this.isPrime(302); // NOT PRIME.

		this.isPrime(303); // NOT PRIME.

		this.isPrime(304); // NOT PRIME.

		this.isPrime(305); // NOT PRIME.

		this.isPrime(306); // NOT PRIME.

		this.isPrime(307); // PRIME.

		this.isPrime(308); // NOT PRIME.

		this.isPrime(309); // NOT PRIME.

		this.isPrime(310); // NOT PRIME.

		this.isPrime(311); // PRIME.

		this.isPrime(312); // NOT PRIME.

		this.isPrime(313); // PRIME.

		this.isPrime(314); // NOT PRIME.

		this.isPrime(315); // NOT PRIME.

		this.isPrime(316); // NOT PRIME.

		this.isPrime(317); // PRIME.

		this.isPrime(318); // NOT PRIME.

		this.isPrime(319); // NOT PRIME.

		this.isPrime(320); // NOT PRIME.

		this.isPrime(321); // NOT PRIME.

		this.isPrime(322); // NOT PRIME.

		this.isPrime(323); // NOT PRIME.

		this.isPrime(324); // NOT PRIME.

		this.isPrime(325); // NOT PRIME.

		this.isPrime(326); // NOT PRIME.

		this.isPrime(327); // NOT PRIME.

		this.isPrime(328); // NOT PRIME.

		this.isPrime(329); // NOT PRIME.

		this.isPrime(330); // NOT PRIME.

		this.isPrime(331); // PRIME.

		this.isPrime(332); // NOT PRIME.

		this.isPrime(333); // NOT PRIME.

		this.isPrime(334); // NOT PRIME.

		this.isPrime(335); // NOT PRIME.

		this.isPrime(336); // NOT PRIME.

		this.isPrime(337); // PRIME.

		this.isPrime(338); // NOT PRIME.

		this.isPrime(339); // NOT PRIME.

		this.isPrime(340); // NOT PRIME.

		this.isPrime(341); // NOT PRIME.

		this.isPrime(342); // NOT PRIME.

		this.isPrime(343); // NOT PRIME.

		this.isPrime(344); // NOT PRIME.

		this.isPrime(345); // NOT PRIME.

		this.isPrime(346); // NOT PRIME.

		this.isPrime(347); // PRIME.

		this.isPrime(348); // NOT PRIME.

		this.isPrime(349); // PRIME.

		this.isPrime(350); // NOT PRIME.

		this.isPrime(351); // NOT PRIME.

		this.isPrime(352); // NOT PRIME.

		this.isPrime(353); // PRIME.

		this.isPrime(354); // NOT PRIME.

		this.isPrime(355); // NOT PRIME.

		this.isPrime(356); // NOT PRIME.

		this.isPrime(357); // NOT PRIME.

		this.isPrime(358); // NOT PRIME.

		this.isPrime(359); // PRIME.

		this.isPrime(360); // NOT PRIME.

		this.isPrime(361); // NOT PRIME.

		this.isPrime(362); // NOT PRIME.

		this.isPrime(363); // NOT PRIME.

		this.isPrime(364); // NOT PRIME.

		this.isPrime(365); // NOT PRIME.

		this.isPrime(366); // NOT PRIME.

		this.isPrime(367); // PRIME.

		this.isPrime(368); // NOT PRIME.

		this.isPrime(369); // NOT PRIME.

		this.isPrime(370); // NOT PRIME.

		this.isPrime(371); // NOT PRIME.

		this.isPrime(372); // NOT PRIME.

		this.isPrime(373); // PRIME.

		this.isPrime(374); // NOT PRIME.

		this.isPrime(375); // NOT PRIME.

		this.isPrime(376); // NOT PRIME.

		this.isPrime(377); // NOT PRIME.

		this.isPrime(378); // NOT PRIME.

		this.isPrime(379); // PRIME.

		this.isPrime(380); // NOT PRIME.

		this.isPrime(381); // NOT PRIME.

		this.isPrime(382); // NOT PRIME.

		this.isPrime(383); // PRIME.

		this.isPrime(384); // NOT PRIME.

		this.isPrime(385); // NOT PRIME.

		this.isPrime(386); // NOT PRIME.

		this.isPrime(387); // NOT PRIME.

		this.isPrime(388); // NOT PRIME.

		this.isPrime(389); // PRIME.

		this.isPrime(390); // NOT PRIME.

		this.isPrime(391); // NOT PRIME.

		this.isPrime(392); // NOT PRIME.

		this.isPrime(393); // NOT PRIME.

		this.isPrime(394); // NOT PRIME.

		this.isPrime(395); // NOT PRIME.

		this.isPrime(396); // NOT PRIME.

		this.isPrime(397); // PRIME.

		this.isPrime(398); // NOT PRIME.

		this.isPrime(399); // NOT PRIME.

		this.isPrime(400); // NOT PRIME.

		this.isPrime(401); // PRIME.

		this.isPrime(402); // NOT PRIME.

		this.isPrime(403); // NOT PRIME.

		this.isPrime(404); // NOT PRIME.

		this.isPrime(405); // NOT PRIME.

		this.isPrime(406); // NOT PRIME.

		this.isPrime(407); // NOT PRIME.

		this.isPrime(408); // NOT PRIME.

		this.isPrime(409); // PRIME.

		this.isPrime(410); // NOT PRIME.

		this.isPrime(411); // NOT PRIME.

		this.isPrime(412); // NOT PRIME.

		this.isPrime(413); // NOT PRIME.

		this.isPrime(414); // NOT PRIME.

		this.isPrime(415); // NOT PRIME.

		this.isPrime(416); // NOT PRIME.

		this.isPrime(417); // NOT PRIME.

		this.isPrime(418); // NOT PRIME.

		this.isPrime(419); // PRIME.

		this.isPrime(420); // NOT PRIME.

		this.isPrime(421); // PRIME.

		this.isPrime(422); // NOT PRIME.

		this.isPrime(423); // NOT PRIME.

		this.isPrime(424); // NOT PRIME.

		this.isPrime(425); // NOT PRIME.

		this.isPrime(426); // NOT PRIME.

		this.isPrime(427); // NOT PRIME.

		this.isPrime(428); // NOT PRIME.

		this.isPrime(429); // NOT PRIME.

		this.isPrime(430); // NOT PRIME.

		this.isPrime(431); // PRIME.

		this.isPrime(432); // NOT PRIME.

		this.isPrime(433); // PRIME.

		this.isPrime(434); // NOT PRIME.

		this.isPrime(435); // NOT PRIME.

		this.isPrime(436); // NOT PRIME.

		this.isPrime(437); // NOT PRIME.

		this.isPrime(438); // NOT PRIME.

		this.isPrime(439); // PRIME.

		this.isPrime(440); // NOT PRIME.

		this.isPrime(441); // NOT PRIME.

		this.isPrime(442); // NOT PRIME.

		this.isPrime(443); // PRIME.

		this.isPrime(444); // NOT PRIME.

		this.isPrime(445); // NOT PRIME.

		this.isPrime(446); // NOT PRIME.

		this.isPrime(447); // NOT PRIME.

		this.isPrime(448); // NOT PRIME.

		this.isPrime(449); // PRIME.

		this.isPrime(450); // NOT PRIME.

		this.isPrime(451); // NOT PRIME.

		this.isPrime(452); // NOT PRIME.

		this.isPrime(453); // NOT PRIME.

		this.isPrime(454); // NOT PRIME.

		this.isPrime(455); // NOT PRIME.

		this.isPrime(456); // NOT PRIME.

		this.isPrime(457); // PRIME.

		this.isPrime(458); // NOT PRIME.

		this.isPrime(459); // NOT PRIME.

		this.isPrime(460); // NOT PRIME.

		this.isPrime(461); // PRIME.

		this.isPrime(462); // NOT PRIME.

		this.isPrime(463); // PRIME.

		this.isPrime(464); // NOT PRIME.

		this.isPrime(465); // NOT PRIME.

		this.isPrime(466); // NOT PRIME.

		this.isPrime(467); // PRIME.

		this.isPrime(468); // NOT PRIME.

		this.isPrime(469); // NOT PRIME.

		this.isPrime(470); // NOT PRIME.

		this.isPrime(471); // NOT PRIME.

		this.isPrime(472); // NOT PRIME.

		this.isPrime(473); // NOT PRIME.

		this.isPrime(474); // NOT PRIME.

		this.isPrime(475); // NOT PRIME.

		this.isPrime(476); // NOT PRIME.

		this.isPrime(477); // NOT PRIME.

		this.isPrime(478); // NOT PRIME.

		this.isPrime(479); // PRIME.

		this.isPrime(480); // NOT PRIME.

		this.isPrime(481); // NOT PRIME.

		this.isPrime(482); // NOT PRIME.

		this.isPrime(483); // NOT PRIME.

		this.isPrime(484); // NOT PRIME.

		this.isPrime(485); // NOT PRIME.

		this.isPrime(486); // NOT PRIME.

		this.isPrime(487); // PRIME.

		this.isPrime(488); // NOT PRIME.

		this.isPrime(489); // NOT PRIME.

		this.isPrime(490); // NOT PRIME.

		this.isPrime(491); // PRIME.

		this.isPrime(492); // NOT PRIME.

		this.isPrime(493); // NOT PRIME.

		this.isPrime(494); // NOT PRIME.

		this.isPrime(495); // NOT PRIME.

		this.isPrime(496); // NOT PRIME.

		this.isPrime(497); // NOT PRIME.

		this.isPrime(498); // NOT PRIME.

		this.isPrime(499); // PRIME.

		this.isPrime(500); // NOT PRIME.

		this.isPrime(501); // NOT PRIME.

		this.isPrime(502); // NOT PRIME.

		this.isPrime(503); // PRIME.

		this.isPrime(504); // NOT PRIME.

		this.isPrime(505); // NOT PRIME.

		this.isPrime(506); // NOT PRIME.

		this.isPrime(507); // NOT PRIME.

		this.isPrime(508); // NOT PRIME.

		this.isPrime(509); // PRIME.

		this.isPrime(510); // NOT PRIME.

		this.isPrime(511); // NOT PRIME.

		this.isPrime(512); // NOT PRIME.

		this.isPrime(513); // NOT PRIME.

		this.isPrime(514); // NOT PRIME.

		this.isPrime(515); // NOT PRIME.

		this.isPrime(516); // NOT PRIME.

		this.isPrime(517); // NOT PRIME.

		this.isPrime(518); // NOT PRIME.

		this.isPrime(519); // NOT PRIME.

		this.isPrime(520); // NOT PRIME.

		this.isPrime(521); // PRIME.

		this.isPrime(522); // NOT PRIME.

		this.isPrime(523); // PRIME.

		this.isPrime(524); // NOT PRIME.

		this.isPrime(525); // NOT PRIME.

		this.isPrime(526); // NOT PRIME.

		this.isPrime(527); // NOT PRIME.

		this.isPrime(528); // NOT PRIME.

		this.isPrime(529); // NOT PRIME.

		this.isPrime(530); // NOT PRIME.

		this.isPrime(531); // NOT PRIME.

		this.isPrime(532); // NOT PRIME.

		this.isPrime(533); // NOT PRIME.

		this.isPrime(534); // NOT PRIME.

		this.isPrime(535); // NOT PRIME.

		this.isPrime(536); // NOT PRIME.

		this.isPrime(537); // NOT PRIME.

		this.isPrime(538); // NOT PRIME.

		this.isPrime(539); // NOT PRIME.

		this.isPrime(540); // NOT PRIME.

		this.isPrime(541); // PRIME.

		this.isPrime(542); // NOT PRIME.

		this.isPrime(543); // NOT PRIME.

		this.isPrime(544); // NOT PRIME.

		this.isPrime(545); // NOT PRIME.

		this.isPrime(546); // NOT PRIME.

		this.isPrime(547); // PRIME.

		this.isPrime(548); // NOT PRIME.

		this.isPrime(549); // NOT PRIME.

		this.isPrime(550); // NOT PRIME.

		this.isPrime(551); // NOT PRIME.

		this.isPrime(552); // NOT PRIME.

		this.isPrime(553); // NOT PRIME.

		this.isPrime(554); // NOT PRIME.

		this.isPrime(555); // NOT PRIME.

		this.isPrime(556); // NOT PRIME.

		this.isPrime(557); // PRIME.

		this.isPrime(558); // NOT PRIME.

		this.isPrime(559); // NOT PRIME.

		this.isPrime(560); // NOT PRIME.

		this.isPrime(561); // NOT PRIME.

		this.isPrime(562); // NOT PRIME.

		this.isPrime(563); // PRIME.

		this.isPrime(564); // NOT PRIME.

		this.isPrime(565); // NOT PRIME.

		this.isPrime(566); // NOT PRIME.

		this.isPrime(567); // NOT PRIME.

		this.isPrime(568); // NOT PRIME.

		this.isPrime(569); // PRIME.

		this.isPrime(570); // NOT PRIME.

		this.isPrime(571); // PRIME.

		this.isPrime(572); // NOT PRIME.

		this.isPrime(573); // NOT PRIME.

		this.isPrime(574); // NOT PRIME.

		this.isPrime(575); // NOT PRIME.

		this.isPrime(576); // NOT PRIME.

		this.isPrime(577); // PRIME.

		this.isPrime(578); // NOT PRIME.

		this.isPrime(579); // NOT PRIME.

		this.isPrime(580); // NOT PRIME.

		this.isPrime(581); // NOT PRIME.

		this.isPrime(582); // NOT PRIME.

		this.isPrime(583); // NOT PRIME.

		this.isPrime(584); // NOT PRIME.

		this.isPrime(585); // NOT PRIME.

		this.isPrime(586); // NOT PRIME.

		this.isPrime(587); // PRIME.

		this.isPrime(588); // NOT PRIME.

		this.isPrime(589); // NOT PRIME.

		this.isPrime(590); // NOT PRIME.

		this.isPrime(591); // NOT PRIME.

		this.isPrime(592); // NOT PRIME.

		this.isPrime(593); // PRIME.

		this.isPrime(594); // NOT PRIME.

		this.isPrime(595); // NOT PRIME.

		this.isPrime(596); // NOT PRIME.

		this.isPrime(597); // NOT PRIME.

		this.isPrime(598); // NOT PRIME.

		this.isPrime(599); // PRIME.

		this.isPrime(600); // NOT PRIME.

		this.isPrime(601); // PRIME.

		this.isPrime(602); // NOT PRIME.

		this.isPrime(603); // NOT PRIME.

		this.isPrime(604); // NOT PRIME.

		this.isPrime(605); // NOT PRIME.

		this.isPrime(606); // NOT PRIME.

		this.isPrime(607); // PRIME.

		this.isPrime(608); // NOT PRIME.

		this.isPrime(609); // NOT PRIME.

		this.isPrime(610); // NOT PRIME.

		this.isPrime(611); // NOT PRIME.

		this.isPrime(612); // NOT PRIME.

		this.isPrime(613); // PRIME.

		this.isPrime(614); // NOT PRIME.

		this.isPrime(615); // NOT PRIME.

		this.isPrime(616); // NOT PRIME.

		this.isPrime(617); // PRIME.

		this.isPrime(618); // NOT PRIME.

		this.isPrime(619); // PRIME.

		this.isPrime(620); // NOT PRIME.

		this.isPrime(621); // NOT PRIME.

		this.isPrime(622); // NOT PRIME.

		this.isPrime(623); // NOT PRIME.

		this.isPrime(624); // NOT PRIME.

		this.isPrime(625); // NOT PRIME.

		this.isPrime(626); // NOT PRIME.

		this.isPrime(627); // NOT PRIME.

		this.isPrime(628); // NOT PRIME.

		this.isPrime(629); // NOT PRIME.

		this.isPrime(630); // NOT PRIME.

		this.isPrime(631); // PRIME.

		this.isPrime(632); // NOT PRIME.

		this.isPrime(633); // NOT PRIME.

		this.isPrime(634); // NOT PRIME.

		this.isPrime(635); // NOT PRIME.

		this.isPrime(636); // NOT PRIME.

		this.isPrime(637); // NOT PRIME.

		this.isPrime(638); // NOT PRIME.

		this.isPrime(639); // NOT PRIME.

		this.isPrime(640); // NOT PRIME.

		this.isPrime(641); // PRIME.

		this.isPrime(642); // NOT PRIME.

		this.isPrime(643); // PRIME.

		this.isPrime(644); // NOT PRIME.

		this.isPrime(645); // NOT PRIME.

		this.isPrime(646); // NOT PRIME.

		this.isPrime(647); // PRIME.

		this.isPrime(648); // NOT PRIME.

		this.isPrime(649); // NOT PRIME.

		this.isPrime(650); // NOT PRIME.

		this.isPrime(651); // NOT PRIME.

		this.isPrime(652); // NOT PRIME.

		this.isPrime(653); // PRIME.

		this.isPrime(654); // NOT PRIME.

		this.isPrime(655); // NOT PRIME.

		this.isPrime(656); // NOT PRIME.

		this.isPrime(657); // NOT PRIME.

		this.isPrime(658); // NOT PRIME.

		this.isPrime(659); // PRIME.

		this.isPrime(660); // NOT PRIME.

		this.isPrime(661); // PRIME.

		this.isPrime(662); // NOT PRIME.

		this.isPrime(663); // NOT PRIME.

		this.isPrime(664); // NOT PRIME.

		this.isPrime(665); // NOT PRIME.

		this.isPrime(666); // NOT PRIME.

		this.isPrime(667); // NOT PRIME.

		this.isPrime(668); // NOT PRIME.

		this.isPrime(669); // NOT PRIME.

		this.isPrime(670); // NOT PRIME.

		this.isPrime(671); // NOT PRIME.

		this.isPrime(672); // NOT PRIME.

		this.isPrime(673); // PRIME.

		this.isPrime(674); // NOT PRIME.

		this.isPrime(675); // NOT PRIME.

		this.isPrime(676); // NOT PRIME.

		this.isPrime(677); // PRIME.

		this.isPrime(678); // NOT PRIME.

		this.isPrime(679); // NOT PRIME.

		this.isPrime(680); // NOT PRIME.

		this.isPrime(681); // NOT PRIME.

		this.isPrime(682); // NOT PRIME.

		this.isPrime(683); // PRIME.

		this.isPrime(684); // NOT PRIME.

		this.isPrime(685); // NOT PRIME.

		this.isPrime(686); // NOT PRIME.

		this.isPrime(687); // NOT PRIME.

		this.isPrime(688); // NOT PRIME.

		this.isPrime(689); // NOT PRIME.

		this.isPrime(690); // NOT PRIME.

		this.isPrime(691); // PRIME.

		this.isPrime(692); // NOT PRIME.

		this.isPrime(693); // NOT PRIME.

		this.isPrime(694); // NOT PRIME.

		this.isPrime(695); // NOT PRIME.

		this.isPrime(696); // NOT PRIME.

		this.isPrime(697); // NOT PRIME.

		this.isPrime(698); // NOT PRIME.

		this.isPrime(699); // NOT PRIME.

		this.isPrime(700); // NOT PRIME.

		this.isPrime(701); // PRIME.

		this.isPrime(702); // NOT PRIME.

		this.isPrime(703); // NOT PRIME.

		this.isPrime(704); // NOT PRIME.

		this.isPrime(705); // NOT PRIME.

		this.isPrime(706); // NOT PRIME.

		this.isPrime(707); // NOT PRIME.

		this.isPrime(708); // NOT PRIME.

		this.isPrime(709); // PRIME.

		this.isPrime(710); // NOT PRIME.

		this.isPrime(711); // NOT PRIME.

		this.isPrime(712); // NOT PRIME.

		this.isPrime(713); // NOT PRIME.

		this.isPrime(714); // NOT PRIME.

		this.isPrime(715); // NOT PRIME.

		this.isPrime(716); // NOT PRIME.

		this.isPrime(717); // NOT PRIME.

		this.isPrime(718); // NOT PRIME.

		this.isPrime(719); // PRIME.

		this.isPrime(720); // NOT PRIME.

		this.isPrime(721); // NOT PRIME.

		this.isPrime(722); // NOT PRIME.

		this.isPrime(723); // NOT PRIME.

		this.isPrime(724); // NOT PRIME.

		this.isPrime(725); // NOT PRIME.

		this.isPrime(726); // NOT PRIME.

		this.isPrime(727); // PRIME.

		this.isPrime(728); // NOT PRIME.

		this.isPrime(729); // NOT PRIME.

		this.isPrime(730); // NOT PRIME.

		this.isPrime(731); // NOT PRIME.

		this.isPrime(732); // NOT PRIME.

		this.isPrime(733); // PRIME.

		this.isPrime(734); // NOT PRIME.

		this.isPrime(735); // NOT PRIME.

		this.isPrime(736); // NOT PRIME.

		this.isPrime(737); // NOT PRIME.

		this.isPrime(738); // NOT PRIME.

		this.isPrime(739); // PRIME.

		this.isPrime(740); // NOT PRIME.

		this.isPrime(741); // NOT PRIME.

		this.isPrime(742); // NOT PRIME.

		this.isPrime(743); // PRIME.

		this.isPrime(744); // NOT PRIME.

		this.isPrime(745); // NOT PRIME.

		this.isPrime(746); // NOT PRIME.

		this.isPrime(747); // NOT PRIME.

		this.isPrime(748); // NOT PRIME.

		this.isPrime(749); // NOT PRIME.

		this.isPrime(750); // NOT PRIME.

		this.isPrime(751); // PRIME.

		this.isPrime(752); // NOT PRIME.

		this.isPrime(753); // NOT PRIME.

		this.isPrime(754); // NOT PRIME.

		this.isPrime(755); // NOT PRIME.

		this.isPrime(756); // NOT PRIME.

		this.isPrime(757); // PRIME.

		this.isPrime(758); // NOT PRIME.

		this.isPrime(759); // NOT PRIME.

		this.isPrime(760); // NOT PRIME.

		this.isPrime(761); // PRIME.

		this.isPrime(762); // NOT PRIME.

		this.isPrime(763); // NOT PRIME.

		this.isPrime(764); // NOT PRIME.

		this.isPrime(765); // NOT PRIME.

		this.isPrime(766); // NOT PRIME.

		this.isPrime(767); // NOT PRIME.

		this.isPrime(768); // NOT PRIME.

		this.isPrime(769); // PRIME.

		this.isPrime(770); // NOT PRIME.

		this.isPrime(771); // NOT PRIME.

		this.isPrime(772); // NOT PRIME.

		this.isPrime(773); // PRIME.

		this.isPrime(774); // NOT PRIME.

		this.isPrime(775); // NOT PRIME.

		this.isPrime(776); // NOT PRIME.

		this.isPrime(777); // NOT PRIME.

		this.isPrime(778); // NOT PRIME.

		this.isPrime(779); // NOT PRIME.

		this.isPrime(780); // NOT PRIME.

		this.isPrime(781); // NOT PRIME.

		this.isPrime(782); // NOT PRIME.

		this.isPrime(783); // NOT PRIME.

		this.isPrime(784); // NOT PRIME.

		this.isPrime(785); // NOT PRIME.

		this.isPrime(786); // NOT PRIME.

		this.isPrime(787); // PRIME.

		this.isPrime(788); // NOT PRIME.

		this.isPrime(789); // NOT PRIME.

		this.isPrime(790); // NOT PRIME.

		this.isPrime(791); // NOT PRIME.

		this.isPrime(792); // NOT PRIME.

		this.isPrime(793); // NOT PRIME.

		this.isPrime(794); // NOT PRIME.

		this.isPrime(795); // NOT PRIME.

		this.isPrime(796); // NOT PRIME.

		this.isPrime(797); // PRIME.

		this.isPrime(798); // NOT PRIME.

		this.isPrime(799); // NOT PRIME.

		this.isPrime(800); // NOT PRIME.

		this.isPrime(801); // NOT PRIME.

		this.isPrime(802); // NOT PRIME.

		this.isPrime(803); // NOT PRIME.

		this.isPrime(804); // NOT PRIME.

		this.isPrime(805); // NOT PRIME.

		this.isPrime(806); // NOT PRIME.

		this.isPrime(807); // NOT PRIME.

		this.isPrime(808); // NOT PRIME.

		this.isPrime(809); // PRIME.

		this.isPrime(810); // NOT PRIME.

		this.isPrime(811); // PRIME.

		this.isPrime(812); // NOT PRIME.

		this.isPrime(813); // NOT PRIME.

		this.isPrime(814); // NOT PRIME.

		this.isPrime(815); // NOT PRIME.

		this.isPrime(816); // NOT PRIME.

		this.isPrime(817); // NOT PRIME.

		this.isPrime(818); // NOT PRIME.

		this.isPrime(819); // NOT PRIME.

		this.isPrime(820); // NOT PRIME.

		this.isPrime(821); // PRIME.

		this.isPrime(822); // NOT PRIME.

		this.isPrime(823); // PRIME.

		this.isPrime(824); // NOT PRIME.

		this.isPrime(825); // NOT PRIME.

		this.isPrime(826); // NOT PRIME.

		this.isPrime(827); // PRIME.

		this.isPrime(828); // NOT PRIME.

		this.isPrime(829); // PRIME.

		this.isPrime(830); // NOT PRIME.

		this.isPrime(831); // NOT PRIME.

		this.isPrime(832); // NOT PRIME.

		this.isPrime(833); // NOT PRIME.

		this.isPrime(834); // NOT PRIME.

		this.isPrime(835); // NOT PRIME.

		this.isPrime(836); // NOT PRIME.

		this.isPrime(837); // NOT PRIME.

		this.isPrime(838); // NOT PRIME.

		this.isPrime(839); // PRIME.

		this.isPrime(840); // NOT PRIME.

		this.isPrime(841); // NOT PRIME.

		this.isPrime(842); // NOT PRIME.

		this.isPrime(843); // NOT PRIME.

		this.isPrime(844); // NOT PRIME.

		this.isPrime(845); // NOT PRIME.

		this.isPrime(846); // NOT PRIME.

		this.isPrime(847); // NOT PRIME.

		this.isPrime(848); // NOT PRIME.

		this.isPrime(849); // NOT PRIME.

		this.isPrime(850); // NOT PRIME.

		this.isPrime(851); // NOT PRIME.

		this.isPrime(852); // NOT PRIME.

		this.isPrime(853); // PRIME.

		this.isPrime(854); // NOT PRIME.

		this.isPrime(855); // NOT PRIME.

		this.isPrime(856); // NOT PRIME.

		this.isPrime(857); // PRIME.

		this.isPrime(858); // NOT PRIME.

		this.isPrime(859); // PRIME.

		this.isPrime(860); // NOT PRIME.

		this.isPrime(861); // NOT PRIME.

		this.isPrime(862); // NOT PRIME.

		this.isPrime(863); // PRIME.

		this.isPrime(864); // NOT PRIME.

		this.isPrime(865); // NOT PRIME.

		this.isPrime(866); // NOT PRIME.

		this.isPrime(867); // NOT PRIME.

		this.isPrime(868); // NOT PRIME.

		this.isPrime(869); // NOT PRIME.

		this.isPrime(870); // NOT PRIME.

		this.isPrime(871); // NOT PRIME.

		this.isPrime(872); // NOT PRIME.

		this.isPrime(873); // NOT PRIME.

		this.isPrime(874); // NOT PRIME.

		this.isPrime(875); // NOT PRIME.

		this.isPrime(876); // NOT PRIME.

		this.isPrime(877); // PRIME.

		this.isPrime(878); // NOT PRIME.

		this.isPrime(879); // NOT PRIME.

		this.isPrime(880); // NOT PRIME.

		this.isPrime(881); // PRIME.

		this.isPrime(882); // NOT PRIME.

		this.isPrime(883); // PRIME.

		this.isPrime(884); // NOT PRIME.

		this.isPrime(885); // NOT PRIME.

		this.isPrime(886); // NOT PRIME.

		this.isPrime(887); // PRIME.

		this.isPrime(888); // NOT PRIME.

		this.isPrime(889); // NOT PRIME.

		this.isPrime(890); // NOT PRIME.

		this.isPrime(891); // NOT PRIME.

		this.isPrime(892); // NOT PRIME.

		this.isPrime(893); // NOT PRIME.

		this.isPrime(894); // NOT PRIME.

		this.isPrime(895); // NOT PRIME.

		this.isPrime(896); // NOT PRIME.

		this.isPrime(897); // NOT PRIME.

		this.isPrime(898); // NOT PRIME.

		this.isPrime(899); // NOT PRIME.

		this.isPrime(900); // NOT PRIME.

		this.isPrime(901); // NOT PRIME.

		this.isPrime(902); // NOT PRIME.

		this.isPrime(903); // NOT PRIME.

		this.isPrime(904); // NOT PRIME.

		this.isPrime(905); // NOT PRIME.

		this.isPrime(906); // NOT PRIME.

		this.isPrime(907); // PRIME.

		this.isPrime(908); // NOT PRIME.

		this.isPrime(909); // NOT PRIME.

		this.isPrime(910); // NOT PRIME.

		this.isPrime(911); // PRIME.

		this.isPrime(912); // NOT PRIME.

		this.isPrime(913); // NOT PRIME.

		this.isPrime(914); // NOT PRIME.

		this.isPrime(915); // NOT PRIME.

		this.isPrime(916); // NOT PRIME.

		this.isPrime(917); // NOT PRIME.

		this.isPrime(918); // NOT PRIME.

		this.isPrime(919); // PRIME.

		this.isPrime(920); // NOT PRIME.

		this.isPrime(921); // NOT PRIME.

		this.isPrime(922); // NOT PRIME.

		this.isPrime(923); // NOT PRIME.

		this.isPrime(924); // NOT PRIME.

		this.isPrime(925); // NOT PRIME.

		this.isPrime(926); // NOT PRIME.

		this.isPrime(927); // NOT PRIME.

		this.isPrime(928); // NOT PRIME.

		this.isPrime(929); // PRIME.

		this.isPrime(930); // NOT PRIME.

		this.isPrime(931); // NOT PRIME.

		this.isPrime(932); // NOT PRIME.

		this.isPrime(933); // NOT PRIME.

		this.isPrime(934); // NOT PRIME.

		this.isPrime(935); // NOT PRIME.

		this.isPrime(936); // NOT PRIME.

		this.isPrime(937); // PRIME.

		this.isPrime(938); // NOT PRIME.

		this.isPrime(939); // NOT PRIME.

		this.isPrime(940); // NOT PRIME.

		this.isPrime(941); // PRIME.

		this.isPrime(942); // NOT PRIME.

		this.isPrime(943); // NOT PRIME.

		this.isPrime(944); // NOT PRIME.

		this.isPrime(945); // NOT PRIME.

		this.isPrime(946); // NOT PRIME.

		this.isPrime(947); // PRIME.

		this.isPrime(948); // NOT PRIME.

		this.isPrime(949); // NOT PRIME.

		this.isPrime(950); // NOT PRIME.

		this.isPrime(951); // NOT PRIME.

		this.isPrime(952); // NOT PRIME.

		this.isPrime(953); // PRIME.

		this.isPrime(954); // NOT PRIME.

		this.isPrime(955); // NOT PRIME.

		this.isPrime(956); // NOT PRIME.

		this.isPrime(957); // NOT PRIME.

		this.isPrime(958); // NOT PRIME.

		this.isPrime(959); // NOT PRIME.

		this.isPrime(960); // NOT PRIME.

		this.isPrime(961); // NOT PRIME.

		this.isPrime(962); // NOT PRIME.

		this.isPrime(963); // NOT PRIME.
		this.isPrime(964); // NOT PRIME.
		this.isPrime(965); // NOT PRIME.
		this.isPrime(966); // NOT PRIME.
		this.isPrime(967); // PRIME.
		this.isPrime(968); // NOT PRIME.
		this.isPrime(969); // NOT PRIME.
		this.isPrime(970); // NOT PRIME.
		this.isPrime(971); // PRIME.
		this.isPrime(972); // NOT PRIME.
		this.isPrime(973); // NOT PRIME.
		this.isPrime(974); // NOT PRIME.
		this.isPrime(975); // NOT PRIME.
		this.isPrime(976); // NOT PRIME.
		this.isPrime(977); // PRIME.
		this.isPrime(978); // NOT PRIME.
		this.isPrime(979); // NOT PRIME.
		this.isPrime(980); // NOT PRIME.
		this.isPrime(981); // NOT PRIME.
		this.isPrime(982); // NOT PRIME.
		this.isPrime(983); // PRIME.
		this.isPrime(984); // NOT PRIME.
		this.isPrime(985); // NOT PRIME.
		this.isPrime(986); // NOT PRIME.
		this.isPrime(987); // NOT PRIME.
		this.isPrime(988); // NOT PRIME.
		this.isPrime(989); // NOT PRIME.
		this.isPrime(990); // NOT PRIME.
		this.isPrime(991); // PRIME.
		this.isPrime(992); // NOT PRIME.
		this.isPrime(993); // NOT PRIME.
		this.isPrime(994); // NOT PRIME.
		this.isPrime(995); // NOT PRIME.
		this.isPrime(996); // NOT PRIME.
		this.isPrime(997); // PRIME.
		this.isPrime(998); // NOT PRIME.
		this.isPrime(999); // NOT PRIME.

	}

}