<html>
<head>
<meta name="viewport" content="width=device-width" />
<title>Control LED with Raspberry Pi using Apache Webserver </title>
</head>
	<style>
	body 
	{
		background-color: rgb(212,250,252);
		font-family: 'Arial';
		
	}
	.red {
		background-color: red;
		width: 18em; height: 4em;
		font-size: 14px;
	}
	.yellow {
		background-color: yellow;
		width: 10em; height: 4em;
		font-size: 20px;
	}
	.green { 
		background-color: green;
		width: 18em; height: 4em;
		font-size: 14px;
	}
	.blue{
		background-color: blue;
		width: 10em; height: 4em;
		font-size: 20px;
	</style>

       <body>
       <center><h1>Control LED from web using Apache Webserver</h1>
         <form method="get" action="index.php">
<h3> Plastic </h3>
 <input class="green" type="submit" value="Turn Green Plastic LED On" name="plast-gon">
            <input class="green" type="submit" value="Turn Green Plastic LED Off" name="plast-goff">
            <br /> <br />  
            <input class ="red" type="submit"  value="Turn Plastic Red LED On" name="plast-ron">
	    <input class=" red" type="submit"  value="Turn Plastic Red LED Off" name="plast-roff">
	    <br /> <br />
	
<hr>

<h3> Food Waste </h3>
            <input class="green" type="submit" value="Turn Food Green LED On" name="food-gon">
            <input class="green" type="submit" value="Turn Food Green LED Off" name="food-goff">
            <br /> <br />  
<input class ="red" type="submit"  value="Turn Food Red LED On" name="food-ron">
            <input class=" red" type="submit"  value="Turn Food Red LED Off" name="food-roff">
            <br /> <br />        
<hr>


<h3> Paper </h3>
<input class="green" type="submit" value="Turn Paper Green LED On" name="papp-gon">
            <input class="green" type="submit" value="Turn Paper Green LED Off" name="papp-goff">
            <br /> <br />  
            <input class ="red" type="submit"  value="Turn Paper Red LED On" name="papp-ron">
            <input class=" red" type="submit"  value="Turn Paper Red LED Off" name="papp-roff">
            <br /> <br />
        
<hr>
</form>

                         </center>
<?php
	shell_exec("gpio -g mode 17 out");
	shell_exec("gpio -g mode 22 out");
	shell_exec("gpio -g mode 27 out");
	shell_exec("gpio -g mode 23 out");
	shell_exec("gpio -g mode 24 out");
	shell_exec("gpio -g mode 25 out");
	
  if(isset($_GET['plast-roff']))
  {
		shell_exec("gpio -g write 17 0");
  }
  else if(isset($_GET['plast-ron']))
  {
    shell_exec("gpio -g write 17 1");
	}
	else if(isset($_GET['plast-gon']))
	{
		shell_exec("gpio -g write 27 1");
	}
	else if(isset($_GET['plast-goff']))
	{
		shell_exec("gpio -g write 27 0");
	}
	else if(isset($_GET['food-gon']))
	{
		shell_exec("gpio -g write 22 1");
	}
	else if(isset($_GET['food-goff']))
	{
		shell_exec("gpio -g write 22 0");
	}
 else if(isset($_GET['food-ron']))
        {
                shell_exec("gpio -g write 23 1");
        }
        else if(isset($_GET['food-roff']))
        {
                shell_exec("gpio -g write 23 0");
        }	
  else if(isset($_GET['papp-gon']))
        {
                shell_exec("gpio -g write 24 1");
        }
        else if(isset($_GET['papp-goff']))
        {
                shell_exec("gpio -g write 24 0");
        }
 else if(isset($_GET['papp-ron']))
        {
                shell_exec("gpio -g write 25 1");
        }
        else if(isset($_GET['papp-roff']))
        {
                shell_exec("gpio -g write 25 0");
        }  

?>
   </body>
</html>

