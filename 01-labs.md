# CE881 Lab 1: Introduction

The aim of this lab is to gain familiarity with using a Java IDE
together with the Android development tools.  You are assumed to be
familiar with Java and Java IDEs in general, and ideally with Eclipse or
Intellij in particular.  If not then seek assistance from the lab or
module supervisor.

![Hello World!](img14.jpg)

The lab will work through various versions of a Hello World program and
introduce some important concepts in the process, finishing off with a
quiz.  Concepts include:

-   Building and running a Hello World program
-   Defining a custom view component
-   Editing XML files to create a GUI
-   Drawing text and setting various Paint parameters
-   Simple event handling

## Hello World

Using your preferred IDE set up an Android project and follow the wizard
to set up a default Hello World app.

I've shown this process below for Intellij 12; other IDEs will differ in
the exact steps and options available and the end result will most
likely look different, but the overall goal will be similar: to
implement and run the simplest possible Android App that says, in some
way, "Hello World".  **Note**: of course the file paths will be
different for you, and the Project SDK may also be different.

![New Android Project](img2.jpg)

Choose the project name and file path (folder) then click next to get to
this dialog box:

![Create Default Activity](img3.jpg)

Accept the defaults by clicking Finish.

You are now ready to run your App.  Do this either on a real Adnroid
device, or using an emulator.  From the tools menu of your IDE start the
emulator.  This will give you a choice of emulators to use: the **Nexus
7** is a reasonable choice, allowing a good size screen view by default.

The emulator takes several minutes to start: use this time to explore
the project that has been created by the IDE.

Find out what the following folders and files are used for: (some are
more obvious than others!).  You can find more details here:
<http://developer.android.com/tools/projects/index.html>

-   assets
-   res
-   gen
-   lib
-   src
-   AndroidManifest.xml
-   main.xml

When the emulator has started it should look something like this:

![Emulator Home Screen](img5.jpg)

When you run the App it might look something like this (top of screen
only is shown)

![Hello World App](img6.jpg)

The auto-generated Activity class is shown below:

![Activity Class Java Code](img6.gif)

Now, using the navigation features of your IDE, find the definition of
**R.layout.main**. 

It should take you to a file called **main.xml** where the layout of the
view is defined.  Here the root view is a container of type
**LinearLayout**, which has a **TextView** with the text set to "Hello
World, My Activity" as its sole child.  When creating views the widths
and heights can be specified in absolute units, but it is more common to
specify them in relation to parent size (e.g. fill\_parent) or to their
content (e.g. wrap\_content). 

**Question:** why do you think the **TextView** has defaulted to this
particular specification of width and height?

![main.xml](img11.jpg)

The **onCreate** method is called just prior to an application starting,
and is the method where the main **View** should be set.  You can change
the **onCreate** method, but be sure that you keep the
**super.onCreate()** method call as the first line.

## Creating a Custom View

Android graphical components subclass the **View** class.  You can write
your own custom view classes to display information in ways most
appropriate for your App.

We're going to take a first look now at how to do this.  The most basic
method to override is **onDraw**.

We'll write a **View** class called **HelloView** that displays the
message "Hello World" in large letters in the middle of the View. 
**HelloView** must extend the **View** class.  It must also have three
constructors that call the **View** superclass constructor.  Your IDE
should be able to auto-create these for you, such that you get this:

![HelloView Skeleton](img8.gif)

Which constructor gets called depends on how the view is invoked.  Let's
label these in order as constructors 1, 2 and 3.  Modify the above
constructors to print a message indicating which one was invoked.

## Android Logcat

Tip: all System.out calls go to the Android Logcat, available as a
window within your IDE.  But LOTS of messages get sent there, so use a
filter to show only the relevant ones.

This may be done by defining a tag String within your class or within
your project, and then prefixing print statements with this.  Android
has a built in Log class which can be used, but you can can also use
standard Java output methods (I prefer the latter approach, since this
will also work when running as stand-alone Java code).

The new tag definition and first constructor is:

![Example printout](img2.gif)

Rewrite the other constructors in a similar way.  Later we'll look at
the Logcat to see which one is invoked.

Now we'll override the **onDraw** method to say HelloWorld in large
letters using some chosen colours.

There are two key classes we need to do this: **Canvas** and **Paint**. 
The Canvas class is like the Graphics class from java.awt, while there
is no direct equivalent to **Paint** in regular Java.  In Android an
object of class **Paint** is used to specify most aspects of how a shape
or string of text is drawn, including:

-   Colour
-   Fill style (e.g. Stroke or Fill)
-   Fill / stroke pattern (e.g. see
    [Shader](http://developer.android.com/reference/android/graphics/Shader.html)
    class)
-   Anti-aliasing
-   Font

One other big difference between Android and awt is that Android uses
**float**s (floating point numbers) to specify coordinates rather than
**int**s.  This is a very welcome change!

Type in the code for the onDraw method as follows.

![onDraw method for HelloView](imgB.gif)

Note that this sample code includes some bad programming style: it
relies on many hard-coded constants.  Don't worry, this is just to show
the basic idea of how to draw text of a particular size and to set paint
colours.  You'll write a much improved version as part of the lab. 

 

We now need to ensure that the view is actually used.  The quickest way
to do this is to modify the onCreate() method of MyActivity as follows:

![Modified onCreate method](imgD.gif)

Now when you run the App it should appear as follows (exact size will
depend on the emulator used)

![Hello View 1](imgA.jpg)

While the app is running look at the Logcat (the exact details will
differ depending on the IDE).  You'll see that with the current version
of the code constructor 1 was invoked:

![Logcat sample](img10.jpg)

**Note:** you can also modify an XML to incorporate the HelloView
component instead of calling the HelloView constructor directly in the
onCreate() method.  To do this revert to the previous onCreate
implementation of MyActivity with the line:

     setContentView(R.layout.main);

but add HelloView as a custom component in the **main.xml** file.  Try
doing this both as a drag and drop operation in the GUI designer, and by
editing the **main.xml** file in text.

For example, you can replace the TextView with a HelloView:

![XML layout with HelloView](img7.gif)

Or simply have HelloView as the root view:

![HelloView as Root](img12.jpg)

**Question**: which constructor is invoked now when you run the App.?

## Programming Exercise

Although the HelloView component sort of works, it's not very nice!  The
exercise is to improve it in the following ways:

1.  Modify the **onDraw** method so that ALL hard-coded constants are
    removed from that method.  Check with a lab assistant that you have
    done this satisfactorily.  They can be declared as static variables
    for now.
2.  Modify the **x** and **y** values so that the text is always drawn
    in the exact centre of **HelloView**.
3.  Modify the font size so that it is always some fraction (e.g. 1/4)
    of the minimum view dimension (i.e. the minimum of the view height
    and width.
4.  Currently the text looks a bit jagged: switch on AntiAlisasing to
    improve this.
5.  Experiment with different Fonts
6.  Experiment with fill effects in Paint objects (by setting up
    Shaders)
7.  Experiment with changing the screen orientation to portrait and to
    landscape, both in the App and on the emulator or device.

The following snip shows the two versions of the text: one is
antialiased, can you tell which?

![Anti Aliasing](HelloAntiAliased.PNG)

**Tip:** it is generally better to draw things with AntiAliasing on:
they will look better.  Many devices will handle the extra processing in
hardware: you won't normally notice any difference in speed.

A version with the text properly centred is shown below:

![Hello Centred Text](img13.jpg)

Hint: to help achieve this you can find the bounding rectangle for the
text you're going to draw using this code within the **onDraw** method
of **HelloView**:

    Rect bounds = new Rect();
    textPaint.getTextBounds(text, 0, text.length(), bounds);

Obviously you should do this AFTER setting the desired text size!

## Flexible Styling

Experiment with more style settings for the text and the background. 
Try to work out a way of setting a theme whereby you can change many
aspects of the style very simply.  For example, can you find a solution
such that changing a single line of code or of XML can completely change
the foreground and background styles?

## Handling Events

We'll now study a simple example of event handling.  When views are
touched, onClick events are generated and sent to them.

There are two main steps to handling events:

1.  Define an implementation of the desired event handling method, in
    this case **onClick**.
2.  Ensure that the object handling the event (i.e. the object that has
    the onClick method) is added as a listener to the View that will be
    the source of these events.

In the example below, I've added an onClick method to the HelloView
class.

 ![onClick method implementation](imgA.gif)

In this very simple example, objects of class HelloView will be both the
source and the destination of the onClick events.  Because they are the
destination we need to declare them as handling these events.  This is
done by declaring that HelloView now implements the appropriate
interface, in this case: View.OnClickListener.

Because **HelloView** is also the source of the click events, we need to
register itself as the listener.  This is done via a call to
**setOnClickListener**.  This is all illustrated as follows:

![Clickable HelloView](imgC.gif)

Now put all this together and check that clicking anywhere on the
HelloView component causes a random colour change e.g.:

![Hello Random World](imgE.jpg)

Finally, this is my garish example of using radial and linear gradient
shaders: (i.e. my solution to part 6 of the programming exercise).

![Gradient Shaders](img15.jpg)

 

## Quiz

You should attempt all the questions in the following quiz.  You can
find some of the answers in the lab work you've already done: some
others you may need to search for.

1.  Which XML file specifies the name of the App?
2.  Which XML file specifies the components and layout of the GUI?  In
    which folder is the file located?
3.  What are the strengths and weaknesses of specifying GUI layouts in
    XML versus in the app's Java code?
4.  What does the "@" specify in an XML attribute?
    (**android:text="@string/hello\_world"**)
5.  When specifying the width or the height of a component what do
    "fill\_parent" and "wrap\_content" mean?
6.  What does DDMS stand for?
7.  What is the origin of the Java code in the **gen** directory, and
    why should you never edit it directly?
8.  Why should you start an AVD (Android Virtual Device) at the start of
    a programming session and then leave the device running?

 

 

